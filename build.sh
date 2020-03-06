#!/usr/env/bin/bash

MPUSIM_PATH=$(pwd)

TENSORFLOW_INSTALL_DIR=$1

if [ -z "${TENSORFLOW_INSTALL_DIR}" ];
    then echo "This script requires the path to a TensorFlow 1.13 installation as parameter"
    exit -1
fi

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

# mpusim_conv2d

mkdir build_mpusim_conv2d_release
cd build_mpusim_conv2d_release

cmake ../mpusim_conv2d -DCMAKE_BUILD_TYPE=Release -DMPUSIM_CONV2D_EIGEN3_LOCAL_INSTALL=ON "-DMPUSIM_CONV2D_EIGEN3_INSTALL_DIR=${MPUSIM_PATH}/build_eigen3_release/install" "-DMPUSIM_CONV2D_MPUSIM_INCLUDE_DIR=${MPUSIM_PATH}/mpu_simulator/include" "-DMPUSIM_CONV2D_MPUSIM_WRAPPER_INCLUDE_DIR=${MPUSIM_PATH}/mpusim_wrapper" "-DMPUSIM_CONV2D_MPUSIM_WRAPPER_INSTALL_DIR=${MPUSIM_PATH}/build_mpusim_wrapper_release" "-DMPUSIM_CONV2D_TENSORFLOW_INSTALL_DIR=${TENSORFLOW_INSTALL_DIR}"
make

cd ..

# mpusim_mat_mul

mkdir build_mpusim_mat_mul_release
cd build_mpusim_mat_mul_release

cmake ../mpusim_fc -DCMAKE_BUILD_TYPE=Release -DMPUSIM_FC_EIGEN3_LOCAL_INSTALL=ON "-DMPUSIM_FC_EIGEN3_INSTALL_DIR=${MPUSIM_PATH}/build_eigen3_release/install" "-DMPUSIM_FC_MPUSIM_INCLUDE_DIR=${MPUSIM_PATH}/mpu_simulator/include" "-DMPUSIM_FC_MPUSIM_WRAPPER_INCLUDE_DIR=${MPUSIM_PATH}/mpusim_wrapper" "-DMPUSIM_FC_MPUSIM_WRAPPER_INSTALL_DIR=${MPUSIM_PATH}/build_mpusim_wrapper_release" "-DMPUSIM_FC_TENSORFLOW_INSTALL_DIR=${TENSORFLOW_INSTALL_DIR}"
make

