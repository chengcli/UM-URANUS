# ===============================================
# Examples published in Cheng et al. (2023), ApJS
# ===============================================
# # cmake .. -DTASK=exo3 -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx

message(STATUS "NetCDF libraries: ${NetCDF_LIBRARIES}")
message(STATUS "NetCDF include dirs: ${NetCDF_INCLUDE_DIRS}")


# 1. Compile dam_break.cpp
if(HYDROSTATIC)
  if(${EQUATION_OF_STATE} STREQUAL "shallow_yz")
    setup_problem(dam_break)
    setup_problem(test_blast)
    setup_problem(W92)
    setup_problem(test_injection)
    setup_problem(steady_zonal)
  endif()
endif()
if(NOT HYDROSTATIC)
  # setup_problem(test_adiabat)
  # setup_problem(hs94)
  # setup_problem(hot_jupiter)
  # setup_problem(hot_jupitertest)
  # setup_problem(polar_dry)
  # setup_problem(uranus2e)
  # setup_problem(uranus2f)
  # setup_problem(uranus2g)
  # setup_problem(uranus2h)
  #setup_problem(uranus2i)
  # setup_problem(uranus2j)
  # setup_problem(uranus2l)
  # setup_problem(uranus2m)
  # setup_problem(uranus2n)
  # setup_problem(uranus2o)
  # setup_problem(uranus2p)
  # setup_problem(uranus2q)
  # setup_problem(uranus2r)
  # setup_problem(uranus2r2)
  setup_problem(uranus2s)
  # setup_problem(uranus2t)
  # setup_problem(uranus2u)
  # setup_problem(uranus2v)
  # setup_problem(uranus2w)
  #setup_problem(rturanus)
  # setup_problemRT(uranusnn)
  # setup_problemRT(uranusnnx)
endif()

# 2. Copy input file to run directory
file(GLOB inputs *.inp *.dat)
foreach(input ${inputs})
    file(COPY ${input} DESTINATION ${CMAKE_BINARY_DIR}/bin)
endforeach()
