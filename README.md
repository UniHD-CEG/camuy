# mpusim

This project provides emulation of computation of convolutional and fully connected layers on weight stationary systolic arrays.

## Dependencies

### Eigen3

Eigen3 is shipped with the project. If the build script build.sh is used, this version is used by default. Usage of a local installation of Eigen3 is possible, and is the default option used in the CMakeList.txt files when building the modules individually without usage of the build script.

### TensorFlow

### Tensorpack

### Other dependencies for building custom TensorFlow operators

## Modules

### mpu_simulator

The actual library emulating matrix multiplication computation on weight stationary systolic arrays.

### mpusim_wrapper

This library serves as a wrapper for the mpu_simulator library. It ensures that only a single instance of the model is active at any given point. It also ensures that the input matrices are padded for input sizes that are outside the limitations of the emulator.

### mpusim_conv2d

This project contains the C++ implementation of a TensorFlow conv2d layer, based on the original TensorFlow Conv2DUsingGemmOp operator found in https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/core/kernels/conv_ops_using_gemm.cc, that uses the mpu_simulator through the mpusim_wrapper library for GEMM-based convolution computation. It also provides a Tensorpack 2d convolution operator based on the on the original Tensorpack Conv2d operator found in https://github.com/tensorpack/tensorpack/blob/master/tensorpack/models/conv2d.py, which calls the custom TensorFlow conv2d operator in the background.

### mpusim_fc

This project implements the C++ implementation of a Tensorflow matrix multiplication layer, based on the TensorFlow operator MatMulOp https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/core/kernels/matmul_op.cc.
