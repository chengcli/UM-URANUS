#!/bin/bash
#SBATCH --job-name=GCMx19_uranus
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=12
#SBATCH --mem-per-cpu=1000m 
#SBATCH --time=2-12:30:00
#SBATCH --account=chengcli1 
#SBATCH --partition=standard
#SBATCH --output=GCM_12_20x.log
module load gcc/10.3.0
module load netcdf-c/4.8.1
module load netcdf-cxx
module load openmpi/5.0.3
module load eigen
mpiexec --mca orte_base_help_aggregate 0 \
        --mca coll ^hcoll \
        --mca pml ob1 \
        --mca btl tcp,self \
        -n 96 uranusnnx.release -i uranusnnx.inp
