#!/usr/env/bin/bash

MPUSIM_PATH=$(pwd)

echo "${MPUSIM_PATH}"

# Eigen3

mkdir build_eigen3_release
cd build_eigen3_release

mkdir install

cmake ../eigen3 -DCMAKE_BUILD_TYPE=Release "-DCMAKE_INSTALL_PREFIX=${MPUSIM_PATH}/build_eigen3_release/install"
make install

cd ..

# mpu_simulator

mkdir build_mpu_simulator_release
cd build_mpu_simulator_release

cmake ../mpu_simulator -DCMAKE_BUILD_TYPE=Release -DMPUSIM_EIGEN3_LOCAL_INSTALL=ON "-DMPUSIM_EIGEN3_INSTALL_DIR=${MPUSIM_PATH}/build_eigen3_release/install"
make

cd ..

# mpusim_wrapper

mkdir build_mpusim_wrapper_release
cd build_mpusim_wrapper_release

cmake ../mpusim_wrapper -DCMAKE_BUILD_TYPE=Release -DMPUSIM_WRAPPER_EIGEN3_LOCAL_INSTALL=ON "-DMPUSIM_WRAPPER_EIGEN3_INSTALL_DIR=${MPUSIM_PATH}/build_eigen3_release/install" "-DMPUSIM_WRAPPER_MPUSIM_INCLUDE_DIR=${MPUSIM_PATH}/mpu_simulator/include" "-DMPUSIM_WRAPPER_MPUSIM_INSTALL_DIR=${MPUSIM_PATH}/build_mpu_simulator_release"
make

cd ..
