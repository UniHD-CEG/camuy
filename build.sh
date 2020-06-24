#!/usr/env/bin/bash

# Copyright (c) 2020 Computing Systems Group
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

MPUSIM_PATH=$(pwd)

TENSORFLOW_INSTALL_DIR=$1

if [ -z "${TENSORFLOW_INSTALL_DIR}" ];
    then echo "This script requires the path to a TensorFlow 1.13 installation as parameter"
    exit -1
fi

# Eigen3

cd third_party
mkdir build_eigen3_release
cd build_eigen3_release

mkdir install

echo "Building eigen3..."

cmake ../eigen3 -DCMAKE_BUILD_TYPE=Release "-DCMAKE_INSTALL_PREFIX=${MPUSIM_PATH}/third_party/build_eigen3_release/install"
make install

cd ../..

# mpu_simulator

mkdir bin
cd bin
mkdir build_mpu_simulator_release
cd build_mpu_simulator_release

echo "Building mpu_simulator..."

cmake ../../mpu_simulator -DCMAKE_BUILD_TYPE=Release -DMPUSIM_EIGEN3_LOCAL_INSTALL=ON "-DMPUSIM_EIGEN3_INSTALL_DIR=${MPUSIM_PATH}/third_party/build_eigen3_release/install"
make

cd ..

# mpusim_wrapper

mkdir build_mpusim_wrapper_release
cd build_mpusim_wrapper_release

echo "Building mpusim_wrapper..."

cmake ../../mpusim_wrapper -DCMAKE_BUILD_TYPE=Release -DMPUSIM_WRAPPER_EIGEN3_LOCAL_INSTALL=ON "-DMPUSIM_WRAPPER_EIGEN3_INSTALL_DIR=${MPUSIM_PATH}/third_party/build_eigen3_release/install" "-DMPUSIM_WRAPPER_MPUSIM_INCLUDE_DIR=${MPUSIM_PATH}/mpu_simulator/include" "-DMPUSIM_WRAPPER_MPUSIM_INSTALL_DIR=${MPUSIM_PATH}/bin/build_mpu_simulator_release"
make

cd ..

# mpusim_conv2d

mkdir build_mpusim_conv2d_release
cd build_mpusim_conv2d_release

echo "Building mpusim_conv2d..."

cmake ../../mpusim_conv2d -DCMAKE_BUILD_TYPE=Release -DMPUSIM_CONV2D_EIGEN3_LOCAL_INSTALL=ON "-DMPUSIM_CONV2D_EIGEN3_INSTALL_DIR=${MPUSIM_PATH}/third_party/build_eigen3_release/install" "-DMPUSIM_CONV2D_MPUSIM_INCLUDE_DIR=${MPUSIM_PATH}/mpu_simulator/include" "-DMPUSIM_CONV2D_MPUSIM_WRAPPER_INCLUDE_DIR=${MPUSIM_PATH}/mpusim_wrapper" "-DMPUSIM_CONV2D_MPUSIM_WRAPPER_INSTALL_DIR=${MPUSIM_PATH}/bin/build_mpusim_wrapper_release" "-DMPUSIM_CONV2D_TENSORFLOW_INSTALL_DIR=${TENSORFLOW_INSTALL_DIR}"
make

cd ..

# mpusim_mat_mul

mkdir build_mpusim_mat_mul_release
cd build_mpusim_mat_mul_release

echo "Building mpusim_mat_mul..."

cmake ../../mpusim_fc -DCMAKE_BUILD_TYPE=Release -DMPUSIM_FC_EIGEN3_LOCAL_INSTALL=ON "-DMPUSIM_FC_EIGEN3_INSTALL_DIR=${MPUSIM_PATH}/third_party/build_eigen3_release/install" "-DMPUSIM_FC_MPUSIM_INCLUDE_DIR=${MPUSIM_PATH}/mpu_simulator/include" "-DMPUSIM_FC_MPUSIM_WRAPPER_INCLUDE_DIR=${MPUSIM_PATH}/mpusim_wrapper" "-DMPUSIM_FC_MPUSIM_WRAPPER_INSTALL_DIR=${MPUSIM_PATH}/bin/build_mpusim_wrapper_release" "-DMPUSIM_FC_TENSORFLOW_INSTALL_DIR=${TENSORFLOW_INSTALL_DIR}"
make

cd ..

