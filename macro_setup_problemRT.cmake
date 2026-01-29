# A small macro used for setting up the build of a problem.
#
# Usage: setup_problem(name)

string(TOLOWER ${CMAKE_BUILD_TYPE} buildl)
string(TOUPPER ${CMAKE_BUILD_TYPE} buildu)

macro(setup_problemRT namel)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_PREFIX_PATH "/home/kevinsh/canoe/canoe/libtorch")
  find_package(Torch CONFIG REQUIRED)
  find_package(NetCDF REQUIRED)

  add_executable(${namel}.${buildl} ${namel}.cpp ${CMAKE_SOURCE_DIR}/main.cpp)

  set_target_properties(
    ${namel}.${buildl}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
               COMPILE_FLAGS ${CMAKE_CXX_FLAGS_${buildu}}
  )

  # Link libraries
  target_link_libraries(${namel}.${buildl}
    PRIVATE
      ${CANOE_LIBRARY_${buildu}}
      ${TORCH_LIBRARIES}
      "/sw/pkgs/arc/stacks/gcc/10.3.0/netcdf-c/4.8.1/lib/libnetcdf.so"
      "/sw/pkgs/arc/stacks/gcc/10.3.0/netcdf-cxx4/4.3.1/lib/libnetcdf_c++4.so"
  )

  # Include directories
  target_include_directories(${namel}.${buildl}
    PRIVATE
    "/sw/pkgs/arc/stacks/gcc/10.3.0/netcdf-c/4.8.1/include"  # NetCDF C headers
    "/sw/pkgs/arc/stacks/gcc/10.3.0/hdf5/1.10.8/include"     # HDF5 headers
    "/sw/pkgs/arc/stacks/gcc/10.3.0/netcdf-cxx4/4.3.1/include" # NetCDF C++ headers
    ${CMAKE_BINARY_DIR}
    ${CANOE_INCLUDE_DIR}
    "/sw/pkgs/arc/eigen/eigen-3.4.0"
    ${MPI_CXX_HEADER_DIR}
    ${MPI_CXX_INCLUDE_PATH}
    ${PNETCDF_INCLUDE_DIR}
    ${TORCH_INCLUDE_DIRS}
  )

  set_property(TARGET ${namel}.${buildl} PROPERTY CXX_STANDARD 17)
endmacro()

# macro(setup_problemRT namel)
#   find_package(NetCDF REQUIRED)
#   set(CMAKE_CXX_STANDARD 17)
#   set(CMAKE_PREFIX_PATH "/home/kevinsh/canoe/canoe/libtorch")
#   find_package(Torch CONFIG REQUIRED)
#   # Backward-compatibility shim for legacy CMake code
#   set(_TORCH_INCLUDES "${TORCH_INCLUDE_DIRS}")
#   set(TORCH_INCLUDE_DIR      "${_TORCH_INCLUDES}")
#   set(TORCH_API_INCLUDE_DIR  "${_TORCH_INCLUDES}")

#   # Libraries (mostly unused if you link Torch::Torch properly)

#   add_executable(${namel}.${buildl} ${namel}.cpp ${CMAKE_SOURCE_DIR}/main.cpp)

#   set_target_properties(
#     ${namel}.${buildl}
#     PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
#                COMPILE_FLAGS ${CMAKE_CXX_FLAGS_${buildu}})

# target_link_libraries(
#   ${namel}.${buildl}
#   PRIVATE
#     ${CANOE_LIBRARY_${buildu}}
#     ${TORCH_LIBRARIES}
#     ${NETCDF_LIBRARIES}         # C library
#     ${NETCDF_CXX_LIBRARY}       # C++ library
#     "/sw/pkgs/arc/stacks/gcc/10.3.0/netcdf-cxx4/4.3.1/lib/libnetcdf_c++4.so"
# )

#   target_include_directories(
#     ${namel}.${buildl}
#     PRIVATE ${CMAKE_BINARY_DIR} ${CANOE_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR}
#             ${MPI_CXX_HEADER_DIR}
#             ${MPI_CXX_INCLUDE_PATH}         ${NETCDF_INCLUDES}          # C headers (from nc-config)
#             ${NETCDF_CXX_INCLUDE_DIR}   # C++ headers
#             "/sw/pkgs/arc/stacks/gcc/10.3.0/netcdf-cxx4/4.3.1/include"
#             ${PNETCDF_INCLUDE_DIR} ${TORCH_API_INCLUDE_DIR})
#   target_include_directories(
#   ${namel}.${buildl}
#   PRIVATE
#     ${NetCDF_INCLUDE_DIRS}
# )
#   set_property(TARGET ${namel}.${buildl} PROPERTY CXX_STANDARD 17)
# endmacro()




