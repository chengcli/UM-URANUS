//========================================================================================                                        
// Athena++ astrophysical MHD code                                                                                                
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code                                                        
// contributors Licensed under the 3-clause BSD License, see LICENSE file for                                                     
// details                                                                                                                        
//========================================================================================                                        
//rt uranus first attempt radiative transfer

// C++ headers                                                                                                                    
#include <cmath>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>

// athena                                                                                                                         
#include <athena/eos/eos.hpp>
#include <athena/field/field.hpp>
#include <athena/hydro/hydro.hpp>
#include <athena/mesh/mesh.hpp>
#include <athena/parameter_input.hpp>


// application                                                                                                                    
#include <application/application.hpp>
#include <application/exceptions.hpp>

// canoe                                                                                                                          
#include <air_parcel.hpp>
#include <configure.hpp>
#include <impl.hpp>

// exo3                                                                                                                           
#include <exo3/cubed_sphere.hpp>
#include <exo3/cubed_sphere_utility.hpp>

#include <snap/thermodynamics/thermodynamics.hpp>
// utils
#include <utils/fileio.hpp>
#include <utils/ndarrays.hpp>
#include <utils/vectorize.hpp>
#include <stdexcept>

//torch
#include <torch/script.h>
#include <torch/torch.h>

#define _sqr(x) ((x) * (x))
#define _qur(x) ((x) * (x) * (x) * (x))
#define _cube(x) ((x) * (x) * (x))

using namespace std;
static Real Rd, cp, T0, Rp, grav, eq_heat_flux,sday,syear, dday, Omega, sigma, emissivity,adtdz,cv,spinupflux,flux_ratio,Tint,initheatdecay,Kt,sponge_tau,spongeheight,conv_time,xHe,tempmean,tempstd,heatthr,heatsf,fluxmean,fluxstd,umumean,umustd;
Real current_time,current_dt;
constexpr double pi = 3.14159265358979323846;
constexpr double P0 = 100000;
std::default_random_engine generator;
std::normal_distribution<double> distribution(0.0, 1.0);                                                                       
/*
"col0": "standard", temp
"col1": "symlog", heating rate
"col2": "standard", lumscale
"col3": "standard", umu
"col4": "log-standard", pressure

*/

Real power(Real base, int exp){
  if (exp == 0) return 1;
  Real result = base;
  for (int k = 1; k < exp; k++){
    result *= base;
  }
  return result;
}
//Day averaged flux is 0.5
//Day-averaged zenith angle and flux
Real normalize_standard(Real x, double mean, double std){
  return (x-mean)/std;
}
torch::Tensor torch_normalize_standard(torch::Tensor x, double mean, double std){
  return (x-mean)/std;
}
torch::Tensor torch_denormalize_symlog(torch::Tensor x,double thr, double sf){
  auto unscaled = x*sf;
  auto abs_unscaled = torch::abs(unscaled);
  auto linear_mask = abs_unscaled <= 1.0;
  auto log_mask    = ~linear_mask;
  auto thr_tensor = torch::full({}, thr, x.options());  // scalar tensor
  auto y = torch::zeros_like(x);
  y.index_put_({linear_mask}, unscaled.index({linear_mask}) * thr_tensor);
  auto log_values = torch::sign(unscaled.index({log_mask}))
                  * thr_tensor
                  * torch::pow(10, abs_unscaled.index({log_mask}) - 1.0);

  y.index_put_({log_mask}, log_values);
  return y;
}



torch::Tensor calcglobal(Real lat, Real time){ //return costheta
//costheta = sin phi sin delta + cos phi cos delta cos h
//time average over day
  Real Ls = fmod(time,2*syear)*2*pi/(syear);
  Real obliquity = 97.77*pi/180;
  Real sindelta = sin(obliquity)*sin(Ls);
  Real cosdelta = std::sqrt(1-sindelta*sindelta);

  if (sin(lat)*sindelta + cos(lat)*cosdelta<=0){//polar night
    return torch::tensor(
    { normalize_standard(0.0, fluxmean, fluxstd),
      normalize_standard(0.5, umumean, umustd) },
    torch::TensorOptions().dtype(torch::kFloat32)
).view({1, 2});
  }else if (sin(lat)*sindelta - cos(lat)*cosdelta>0){ //polar day
    Real avgcostheta = sin(lat)*sindelta;
    return torch::tensor(
      { normalize_standard(1.0, fluxmean, fluxstd),
        normalize_standard(avgcostheta, umumean, umustd) },
      torch::TensorOptions().dtype(torch::kFloat32)
  ).view({1, 2});
  }else{
    double arg = -sin(lat)*sindelta / (cos(lat)*cosdelta);
    arg = std::min(1.0, std::max(-1.0, arg));
    Real hsunset = acos(arg);
    //integrate to get average costheta
    //avg = (sin phi sin delta*hsunset + cos phi cos delta sin hsunset)/pi
    Real avgcostheta = (sin(lat)*sindelta*hsunset +cos(lat)*cosdelta*sin(hsunset))/pi;
    return torch::tensor(
        { normalize_standard(hsunset/pi, fluxmean, fluxstd),
          normalize_standard(avgcostheta, umumean, umustd) },
        torch::TensorOptions().dtype(torch::kFloat32)
    ).view({1, 2});
  }
}// average when flux is nonzero and umu is positive?




torch::Tensor regrid(const std::vector<double>& x, //x is flexible in size
                     const std::vector<double>& y, //y is flexible in size
                     const std::vector<double>& xq){ //xq has size 100
    std::vector<double> result(256);
    for (int j = 0; j < 256; ++j){
      // Clamp to range
      if (j>=100){
        result[j] = -9999.0; //padding
      }else if (xq[j] >= x.front()){
        result[j]= y.front();
      }else if (xq[j] <= x.back()){
        result[j]= y.back();
      }else{
        auto it = std::upper_bound(x.begin(), x.end(), xq[j],
                                  std::greater<double>()); // use > for decreasing        
        size_t i = std::distance(x.begin(), it);
        if (i == 0) {
            i = 0;
        } else {
            i = i - 1;
        }
        // Clamp to avoid going past the last element
        i = std::min(i, x.size() - 2);
        // Linear interpolation
        result[j] = y[i] + (y[i+1] - y[i])*(xq[j] - x[i]) / (x[i+1] - x[i]);
        // std::cout<<"x[i] xq[j] x[i+1]"<<x[i]<<" "<<xq[j]<<" "<<x[i+1]<<std::endl;
      }
      // std::cout<<y[j]<<"seq seqinterp"<<result[j]<<std::endl;
    }    
  return torch_normalize_standard(
      torch::tensor(
          result,
          torch::TensorOptions().dtype(torch::kFloat32)
      ).view({1, 256, 1}),
      tempmean,
      tempstd
  );
}

std::vector<double> degrid(const std::vector<double>& x, const torch::Tensor normy, const std::vector<double>& xq){
  std::vector<double> result(100);
  torch::Tensor ytens = torch_denormalize_symlog(normy,heatthr,heatsf);
  ytens = ytens.contiguous().to(torch::kFloat64);

  // Get the number of elements
  auto numel = ytens.numel();

  // Create a std::vector and copy data
  std::vector<double> y(numel);
  std::memcpy(y.data(), ytens.data_ptr<double>(), numel * sizeof(double));

  for (int j = 0; j < 100; ++j){
    // Clamp to range 
    if (xq[j] >= x.front()){
      result[j]= y.front();
    }else if (xq[j] <= x.back()){
      result[j]= y.back();
    }else{
      auto it = std::upper_bound(x.begin(), x.end(), xq[j],
                                std::greater<double>()); // use > for decreasing        
      size_t i = std::distance(x.begin(), it);
      if (i == 0) {
          i = 0;
      } else {
          i = i - 1;
      }
      // Clamp to avoid going past the last element
      i = std::min(i, x.size() - 2);
      // Linear interpolation
      result[j] = y[i] + (y[i+1] - y[i])*(xq[j] - x[i]) / (x[i+1] - x[i]);
      // std::cout<<"x[i] xq[j] x[i+1]"<<x[i]<<" "<<xq[j]<<" "<<x[i+1]<<std::endl;
    }
  }
  return result;    
}

void Forcing(MeshBlock *pmb, Real const time, Real const dt,
             AthenaArray<Real> const &w, const AthenaArray<Real> &prim_scalar,
             AthenaArray<Real> const &bcc, AthenaArray<Real> &du,
             AthenaArray<Real> &cons_scalar) {
  
  auto pexo3 = pmb->pimpl->pexo3;
  current_time = time;
  Real om_earth = Omega;
  
  for (int k = pmb->ks; k <= pmb->ke; ++k)
    for (int j = pmb->js; j <= pmb->je; ++j)
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        Real lat, lon;
        pexo3->GetLatLon(&lat, &lon, k, j, i);
        // coriolis force                                                                                                         
        Real f = 2. * om_earth * sin(lat);
        Real f2 = 2. * om_earth * cos(lat);
        Real U, V;

        pexo3->GetUV(&U, &V, w(IVY, k, j, i), w(IVZ, k, j, i), k, j, i);

        Real m1 = w(IDN, k, j, i) * w(IVX, k, j, i);
        Real m2 = w(IDN, k, j, i) * U;
        Real m3 = w(IDN, k, j, i) * V;

        // du(IM1, k, j, i) += dt * f * m2;                                                                                       
        Real ll_acc_U = f * m3;  //- f2 * m1;                                                                                     
        Real ll_acc_V = -f * m2;
        Real acc1, acc2, acc3;
        pexo3->GetVyVz(&acc2, &acc3, ll_acc_U, ll_acc_V, k, j, i);
        pexo3->ContravariantVectorToCovariant(j, k, acc2, acc3, &acc2, &acc3);
        du(IM2, k, j, i) += dt * acc2;
        du(IM3, k, j, i) += dt * acc3;
      } 

    // Heating   
  torch::jit::script::Module module = torch::jit::load("best_model_12_19.pt");
  module.eval(); //lower bound of RT is 4 bar


  current_dt = dt;
  Real boostrad = std::fmax(1.0,5e5*exp(time/(-1e4)));
  std::vector<double> basepress = { 474464.0, 444378.0, 416200.0, 389808.0, 365090.0, 341939.0, 320256.0, 299948.0, 280928.0, 263114.0, 246430.0, 230804.0, 216168.0, 202461.0, 189622.0, 177598.0, 166337.0, 155789.0, 145910.0, 136658.0, 127992.0, 119876.0, 112275.0, 105155.0, 98487.2, 92242.1, 86392.9, 80914.6, 75783.7, 70978.2, 66477.4, 62262.0, 58313.9, 54616.2, 51152.9, 47909.2, 44871.3, 42025.9, 39361.0, 36865.1, 34527.4, 32338.0, 30287.4, 28366.9, 26568.1, 24883.4, 23305.5, 21827.7, 20443.6, 19147.2, 17933.1, 16795.9, 15730.9, 14733.4, 13799.1, 12924.1, 12104.5, 11337.0, 10618.1, 9944.8, 9314.2, 8723.6, 8170.4, 7652.3, 7167.1, 6712.6, 6286.9, 5888.3, 5514.9, 5165.2, 4837.7, 4530.9, 4243.6, 3974.5, 3722.5, 3486.4, 3265.3, 3058.3, 2864.4, 2682.7, 2512.6, 2353.3, 2204.1, 2064.3, 1933.4, 1810.8, 1696.0, 1588.4, 1487.7, 1393.4, 1305.0, 1222.3, 1144.8, 1072.2, 1004.2, 940.5, 880.9, 825.0, 772.7, 723.7 };
  std::vector<double> regridtemp(100);
  std::vector<double> pressvec(pmb->ie - pmb->is + 1);
  std::vector<double> tempvec(pmb->ie - pmb->is + 1);
  std::vector<double> heating(pmb->ie - pmb->is + 1); // kelvin/sec
  for (int k = pmb->ks; k <= pmb->ke; ++k) //last index is vertical
      for (int j = pmb->js; j <= pmb->je; ++j) {
        for (int i = pmb->is; i <= pmb->ie; ++i){
          pressvec[i-pmb->is] = pmb->phydro->w(IPR, k, j,i);
          tempvec[i-pmb->is] = pmb->phydro->w(IPR, k, j,i) / pmb->phydro->w(IDN, k, j,i) / Rd;
        }
        //interpolate and normalize
        torch::Tensor regridtemp = regrid(pressvec,tempvec,basepress);//normalize and regrid
        Real lat, lon;
        pexo3->GetLatLon(&lat, &lon, k, j, pmb->is);
        auto mask = torch::ones({1, 256}, torch::kBool);
        mask.index_put_({0, torch::indexing::Slice(0, 100)}, false);
        torch::Tensor global_features = calcglobal(lat,time); //normalized global features
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(regridtemp);           // required
        inputs.push_back(global_features);    // optional global_features
        inputs.push_back(mask);

        // Forward pass
        torch::Tensor output = module.forward(inputs).toTensor();
        //de normalize and deinterpolate
        heating = degrid(basepress,output,pressvec);//pressure decreases, heigh increases with increasing index          
        for (int i = pmb->is; i <= pmb->ie; ++i){
          if ((pmb->phydro->w(IPR, k, j,i) / pmb->phydro->w(IDN, k, j,i) / Rd) >= 30){
            du(IEN, k, j, i) +=dt*boostrad*(cp - Rd)*w(IDN, k, j, i)*heating[i-pmb->is];
          }
        }
        // std::cout<<heating[0]<<" "<<heating[1]<<" "<<heating[2]<<std::endl;
      }       
                                                              
}

Real AngularMomentum(MeshBlock *pmb, int iout) {
  auto pexo3 = pmb->pimpl->pexo3;
  Real AMz = 0;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks,
      ke = pmb->ke;

  for (int k = ks; k <= ke; k++) {
    for (int j = js; j <= je; j++) {
      for (int i = is; i <= ie; i++) {
        Real x1l = pmb->pcoord->x1f(i);
        Real x1u = pmb->pcoord->x1f(i + 1);
        Real U, V;
        Real lat, lon;
        pexo3->GetLatLon(&lat, &lon, k, j, i);
        pexo3->GetUV(&U, &V, pmb->phydro->w(IVY, k, j, i),
                     pmb->phydro->w(IVZ, k, j, i), k, j, i);

        Real xt = tan(pmb->pcoord->x2v(j));
        Real yt = tan(pmb->pcoord->x3v(k));
        Real sin_theta =
            sqrt((1.0 + xt * xt + yt * yt) / (1.0 + xt * xt) / (1.0 + yt * yt));

        Real x1 = tan(pmb->pcoord->x2f(j));
        Real x2 = tan(pmb->pcoord->x2f(j + 1));
        Real y = tan(pmb->pcoord->x3v(k));
        Real delta1 = sqrt(1.0 + x1 * x1 + y * y);
        Real delta2 = sqrt(1.0 + x2 * x2 + y * y);
        Real dx2_ang = acos(1 / (delta1 * delta2) * (1 + x1 * x2 + y * y));

        Real x = tan(pmb->pcoord->x2v(j));
        Real y1 = tan(pmb->pcoord->x3f(k));
        Real y2 = tan(pmb->pcoord->x3f(k + 1));
        delta1 = sqrt(1.0 + x * x + y1 * y1);
        delta2 = sqrt(1.0 + x * x + y2 * y2);
        Real dx3_ang = acos(1 / (delta1 * delta2) * (1 + x * x + y1 * y2));

        Real vol = pmb->pcoord->dx1f(i) * dx2_ang * dx3_ang * sin_theta;

        // Originally here used cos(lat), which is x2v-pi, strange                                                                
        AMz += pmb->phydro->w(IDN, k, j, i) * vol *
               sqrt((_sqr(x1l) + _sqr(x1u)) / 2.) * cos(lat) *
               (Omega * sqrt(0.5 * (_sqr(x1l) + _sqr(x1u))) * cos(lat) + U);
      }
    }
  }

  return AMz;
}

//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)                                                                          
void Mesh::InitUserMeshData(ParameterInput *pin) {
  Real day_to_s = 8.64E4;
  // forcing parameters                                                                                                           
  // thermodynamic parameters                                                                                                     
  Real gamma = pin->GetReal("hydro", "gamma");
  grav = -pin->GetReal("hydro", "grav_acc1");
  // Ts = pin->GetReal("problem", "Ts");
  Rd = pin->GetReal("thermodynamics", "Rd");
  // eq_heat_flux = pin->GetReal("problem", "eq_heat_flux");
  sday = pin->GetReal("problem", "sday");
  syear = pin->GetReal("problem", "syear");
  Tint = pin->GetReal("problem", "Tint");
  cp = gamma / (gamma - 1.) * Rd;
  cv = cp/gamma;
  dday = syear/(1+syear/sday); //diurnal day inseconds
  Omega = -2*pi/sday; //retrograde
  sigma = 5.670374419E-8;
  // flux_ratio = 0.25*1.14*eq_heat_flux/(sigma*_qur(Ts));
  // int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  // Real currentforcing[ke-ks][je-js][ie-is] = {};
  tempmean = pin->GetReal("problem", "tempmean"); //specific cooling/heating rate
  tempstd = pin->GetReal("problem", "tempstd"); //specific cooling/heating rate
  heatthr = pin->GetReal("problem", "heatthr"); //specific cooling/heating rate
  heatsf = pin->GetReal("problem", "heatsf"); //specific cooling/heating rate
  umumean = pin->GetReal("problem", "umumean"); //specific cooling/heating rate
  umustd = pin->GetReal("problem", "umustd"); //specific cooling/heating rate
  fluxmean = pin->GetReal("problem", "fluxmean"); //specific cooling/heating rate
  fluxstd = pin->GetReal("problem", "fluxstd"); //specific cooling/heating rate

  // forcing function                                                                                                             
  EnrollUserExplicitSourceFunction(Forcing);
  AllocateUserHistoryOutput(1);
  EnrollUserHistoryOutput(0, AngularMomentum, "z-angular-mom");
}

//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)                                                                     
//  \brief Held-Suarez problem generator                               
void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  MeshBlock *pmb = this; // optional, for clarity
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks,ke = pmb->ke;
  auto pexo3 = pimpl->pexo3;
  Real grav = -pin->GetReal("hydro", "grav_acc1");
  Real gamma = pin->GetReal("hydro", "gamma");
  // Ts = pin->GetReal("problem", "Ts");
  Rd = pin->GetReal("thermodynamics", "Rd");
  cp = gamma / (gamma - 1.) * Rd;
  Rp = pin->GetReal("problem", "Rp");
  // emissivity = pin->GetReal("problem", "emissivity");
  // eq_heat_flux = pin->GetReal("problem", "eq_heat_flux");
  adtdz = pin->GetReal("problem", "adtdz");
  // spinupflux = pin->GetReal("problem", "spinupflux");
  // initheatdecay = pin->GetReal("problem", "initheatdecay");
  sponge_tau = pin->GetReal("problem", "sponge_tau");
  spongeheight = pin->GetReal("problem", "spongeheight");
  cp = gamma / (gamma - 1.) * Rd;
  dday = syear/(1+syear/sday); //diurnal day inseconds
  Omega = -2*pi/sday; //retrograde
  sigma = 5.670374419E-8;
  tempmean = pin->GetReal("problem", "tempmean"); //specific cooling/heating rate
  tempstd = pin->GetReal("problem", "tempstd"); //specific cooling/heating rate
  heatthr = pin->GetReal("problem", "heatthr"); //specific cooling/heating rate
  heatsf = pin->GetReal("problem", "heatsf"); //specific cooling/heating rate
  umumean = pin->GetReal("problem", "umumean"); //specific cooling/heating rate
  umustd = pin->GetReal("problem", "umustd"); //specific cooling/heating rate
  fluxmean = pin->GetReal("problem", "fluxmean"); //specific cooling/heating rate
  fluxstd = pin->GetReal("problem", "fluxstd"); //specific cooling/heating rate
  string out2file = pin->GetString("problem","out2");
  string out3file = pin->GetString("problem","out3");

/*out2
rho      (time, x1, x2, x3) float32 1MB ...
press    (time, x1, x2, x3) float32 1MB ...
vel3     (time, x1, x2, x3) float32 1MB ...
vel1     (time, x1, x2, x3) float32 1MB ...
vel2     (time,x1, x2, x3) float32 1MB ...


*/
// Variables to read
  // Variables to read
  Mesh *pm = pmb->pmy_mesh;

  int Nx = pm->mesh_size.nx1;
  int Ny = pm->mesh_size.nx2;
  int Nz = pm->mesh_size.nx3;
  const int nx = ie - is + 1;
  const int ny = je - js + 1;
  const int nz = ke - ks + 1;

  const size_t ncell = static_cast<size_t>(26) * 96*144;//total cells in mesh, less than total cells in data

  // -----------------------------
  // Allocate buffers
  // -----------------------------
  std::vector<float> rho(ncell);
  std::vector<float> press(ncell);
  std::vector<float> vel1(ncell);
  std::vector<float> vel2(ncell);
  std::vector<float> vel3(ncell);

  // -----------------------------
  // Helper to read one file
  // -----------------------------
  auto read_bin = [&](const std::string &fname, std::vector<float> &buf) {
    std::ifstream fin(fname, std::ios::binary);
    if (!fin) {
      throw std::runtime_error("Failed to open " + fname);
    }
    fin.read(reinterpret_cast<char*>(buf.data()),
            buf.size() * sizeof(float));
    if (!fin) {
      throw std::runtime_error("Failed to read " + fname);
    }
  };

  // -----------------------------
  // Read files
  // -----------------------------
  read_bin("rho.bin",   rho);
  read_bin("press.bin", press);
  read_bin("vel1.bin",  vel1);
  read_bin("vel2.bin",  vel2);
  read_bin("vel3.bin",  vel3);

  // Debug sanity check
  if (Globals::my_rank == 0) {
    std::cout << "IC check rho[0]=" << rho[0]
              << " vel1[0]=" << vel1[0] << std::endl;
  }
  int nx_b = ie - is + 1;   // block size in x (physical cells)
  int ny_b = je - js + 1;
  int nz_b = ke - ks + 1;
  // -----------------------------
  // Map to Athena++ arrays
  // Layout: idx = i + j*nx + k*nx*ny
  size_t idx = is + nx*(js + ny*ks);  // matches NumPy C-order
  std::cout<<"is + nx*(js + ny*ks)"<<rho[idx]<<std::endl;
  std::cout<<"0 index"<<rho[0]<<std::endl;
  std::cout<< "is ie = " << is << " " << ie << "\n"<< "js je = " << js << " " << je << "\n"<< "ks ke = " << ks << " " << ke << std::endl;
  // -----------------------------
  int off_i = pmb->loc.lx1 * nx_b;
  int off_j = pmb->loc.lx2 * ny_b;
  int off_k = pmb->loc.lx3 * nz_b;
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        // local (0-based) index inside block
        int ii = i - is;
        int jj = j - js;
        int kk = k - ks;

        // global (0-based) index in full domain
        int ig = off_i + ii+4;
        int jg = off_j + jj;
        int kg = off_k + kk;

        // flatten global array written by NumPy (C-order)
        size_t idx = kg + Nz * (jg + Ny * ig);

        const Real r  = rho[idx];
        const Real vx = vel1[idx];
        const Real vy = vel2[idx];
        const Real vz = vel3[idx];
        const Real p  = press[idx];

        // Primitive
        pmb->phydro->w(IDN, k, j, i) = r;
        pmb->phydro->w(IVX, k, j, i) = vx;
        pmb->phydro->w(IVY, k, j, i) = vy;
        pmb->phydro->w(IVZ, k, j, i) = vz;
        pmb->phydro->w(IPR, k, j, i) = p;

        // Conserved
        pmb->phydro->u(IDN, k, j, i) = r;
        pmb->phydro->u(IM1, k, j, i) = r * vx;
        pmb->phydro->u(IM2, k, j, i) = r * vy;
        pmb->phydro->u(IM3, k, j, i) = r * vz;

        const Real ek = 0.5 * r * (vx*vx + vy*vy + vz*vz);
        pmb->phydro->u(IEN, k, j, i) = p / (gamma - 1.0) + ek;
      }
    }
  }
  pmb->phydro->hbvar.SwapHydroQuantity(pmb->phydro->w, HydroBoundaryQuantity::prim);
  pmb->pbval->ApplyPhysicalBoundaries(0., 0., pmb->pbval->bvars_main_int);
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  AllocateUserOutputVariables(7);
  SetUserOutputVariableName(0, "temp");
  SetUserOutputVariableName(1, "theta");
  SetUserOutputVariableName(2, "lat");
  SetUserOutputVariableName(3, "lon");
  SetUserOutputVariableName(4, "vlat");
  SetUserOutputVariableName(5, "vlon");
  SetUserOutputVariableName(6, "dTdt");//change in temperature per timestep
}

  // \brif Output distributions of temperature and potential temperature.                                                           
void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  MeshBlock *pmb = this; // optional, for clarity
  auto pexo3 = pimpl->pexo3;
  auto pthermo = Thermodynamics::GetInstance();
      // Heating   
  torch::jit::script::Module module = torch::jit::load("best_model_12_19.pt");
  module.eval(); //lower bound of RT is 4 bar

  Real time = pmb->pmy_mesh->time;
  Real boostrad = std::fmax(1.0,5e5*exp(time/(-5e3)));
  //NN good for pressures 4e5 Pa to 8e3 Pa
  std::vector<double> basepress = { 474464.0, 444378.0, 416200.0, 389808.0, 365090.0, 341939.0, 320256.0, 299948.0, 280928.0, 263114.0, 246430.0, 230804.0, 216168.0, 202461.0, 189622.0, 177598.0, 166337.0, 155789.0, 145910.0, 136658.0, 127992.0, 119876.0, 112275.0, 105155.0, 98487.2, 92242.1, 86392.9, 80914.6, 75783.7, 70978.2, 66477.4, 62262.0, 58313.9, 54616.2, 51152.9, 47909.2, 44871.3, 42025.9, 39361.0, 36865.1, 34527.4, 32338.0, 30287.4, 28366.9, 26568.1, 24883.4, 23305.5, 21827.7, 20443.6, 19147.2, 17933.1, 16795.9, 15730.9, 14733.4, 13799.1, 12924.1, 12104.5, 11337.0, 10618.1, 9944.8, 9314.2, 8723.6, 8170.4, 7652.3, 7167.1, 6712.6, 6286.9, 5888.3, 5514.9, 5165.2, 4837.7, 4530.9, 4243.6, 3974.5, 3722.5, 3486.4, 3265.3, 3058.3, 2864.4, 2682.7, 2512.6, 2353.3, 2204.1, 2064.3, 1933.4, 1810.8, 1696.0, 1588.4, 1487.7, 1393.4, 1305.0, 1222.3, 1144.8, 1072.2, 1004.2, 940.5, 880.9, 825.0, 772.7, 723.7 };
  std::vector<double> regridtemp(100);
  std::vector<double> pressvec(pmb->ie - pmb->is + 1);
  std::vector<double> tempvec(pmb->ie - pmb->is + 1);
  std::vector<double> heating(pmb->ie - pmb->is + 1); // kelvin/sec
            
  
  for (int k = ks; k <= ke; ++k)
    for (int j = js; j <= je; ++j)
      for (int i = is; i <= ie; ++i) {
        Real prim[NHYDRO];
        for (int n = 0; n < NHYDRO; ++n) prim[n] = phydro->w(n, j, i);
        Real temp = phydro->w(IPR, k, j, i) / phydro->w(IDN, k, j, i) / Rd; //assume hydrostatic??
        user_out_var(0, k, j, i) = temp;
        user_out_var(1, k, j, i) =
            temp * pow(P0 / phydro->w(IPR, k, j, i), Rd / cp);
        Real lat, lon;
        Real U, V;
        pexo3->GetLatLon(&lat, &lon, k, j, i);
        pexo3->GetUV(&U, &V, phydro->w(IVY, k, j, i), phydro->w(IVZ, k, j, i),
                     k, j, i);
        user_out_var(2, k, j, i) = lat;
        user_out_var(3, k, j, i) = lon;
        user_out_var(4, k, j, i) = U;
        user_out_var(5, k, j, i) = V;
      }
  for (int k = pmb->ks; k <= pmb->ke; ++k) //last index is vertical
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i){
        pressvec[i-pmb->is] = pmb->phydro->w(IPR, k, j,i);
        tempvec[i-pmb->is] = pmb->phydro->w(IPR, k, j,i) / pmb->phydro->w(IDN, k, j,i) / Rd;
      }
      //interpolate and normalize
      torch::Tensor regridtemp = regrid(pressvec,tempvec,basepress);//normalize and regrid
      Real lat, lon;
      pexo3->GetLatLon(&lat, &lon, k, j, pmb->is);
      auto mask = torch::ones({1, 256}, torch::kBool);
      mask.index_put_({0, torch::indexing::Slice(0, 100)}, false);
      torch::Tensor global_features = calcglobal(lat,time); //normalized global features
      std::vector<torch::jit::IValue> inputs;
      inputs.push_back(regridtemp);           // required
      inputs.push_back(global_features);    // optional global_features
      inputs.push_back(mask);
      // Forward pass
      torch::Tensor output = module.forward(inputs).toTensor();
      //de normalize and deinterpolate
      heating = degrid(basepress,output,pressvec);//pressure decreases, heigh increases with increasing index          
      for (int i = pmb->is; i <= pmb->ie; ++i){
        if (user_out_var(0, k, j, i ) >= 30){
          user_out_var(6, k, j, i) =boostrad*heating[i-pmb->is];
        }else{
          user_out_var(6, k, j, i) = 0;
        }
      }
      // std::cout<<heating[0]<<" "<<heating[1]<<" "<<heating[2]<<std::endl;
    }  
}

