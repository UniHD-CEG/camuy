# mpusim

This project provides emulation of computation of convolutional and fully connected layers on weight stationary systolic arrays. It was developed as part of the master thesis "Efficient Design and Mapping of Deep Neural Networks onto Fixed Processing Units". It consists of the following four modules:

## Modules

### mpu_simulator

The actual library emulating matrix multiplication computation on weight stationary systolic arrays.
If you want to use the project itself without the wrapper and the TensorFlow/Tensorpack operators, the usage is as follows:

#### Instantiate a MatrixProcessingUnit object with the desired weight, activation, and accumulator/result data type, systolic array size, activation FIFO depth, accumulator array height, and maximum unified buffer size
Example:

    MatrixProcessingUnit<WeightDatatype, ActivationDatatype, AccumulatorDatatype> matrixProcessingUnit(systolicArrayWidth,
                                                                            systolicArrayHeight,
                                                                            activationFifoDepth,
                                                                            accumulatorArrayHeight,
                                                                            unifiedBufferSizeByte);

#### Set the desired debug output verbosity
The tool implements two debug output verbosity levels. Debug messages are activated by passing `true` to the method `setDebugFlag()`. Verbose debug output can be activated by passing `true` to the method `setDebugOutputVerboseFlag()`.

#### Optional: Register a callback to collect execution metrics
Using the method `registerLogEntryAvailableCallback()`, a callback function can be registered that will be called upon successful matrix multiplication computation. This callback contains execution metrics such as total required iterations and data movements between the functional units and the unified buffer of the emulated MPU.
The `mpuStatisticsLogger` class contained in the mpu_simulator project provides such a callback function that simply stores the received metrics in a CSV file, with a filename generated from the combined directory and name prefix and a weight/activation/accumulator datatype size combination.
Example:

    MpuStatisticsLogger mpuStatisticsLogger("test", sizeof(WeightDatatype),
                                                sizeof(ActivationDatatype),
                                                sizeof(AccumulatorDatatype));

    matrixProcessingUnit.registerLogEntryAvailableCallback(
                            [&mpuStatisticsLogger](MpuStatisticsLogEntry&& mpuStatisticsLogEntry){
        mpuStatisticsLogger.addMpuStatisticsLogEntry(std::move(mpuStatisticsLogEntry));

### mpusim_wrapper

This library serves as a wrapper for the mpu_simulator library. It ensures that only a single instance of the model is active at any given point. It also ensures that the input matrices are padded for input sizes that are outside the limitations of the emulator.

### mpusim_conv2d

This project contains the C++ implementation of a TensorFlow conv2d layer, based on the original TensorFlow Conv2DUsingGemmOp operator found in https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/core/kernels/conv_ops_using_gemm.cc, that uses the mpu_simulator through the mpusim_wrapper library for GEMM-based convolution computation. It also provides a Tensorpack 2d convolution operator based on the on the original Tensorpack Conv2d operator found in https://github.com/tensorpack/tensorpack/blob/master/tensorpack/models/conv2d.py, which calls the custom TensorFlow conv2d operator in the background.

### mpusim_fc

This project implements the C++ implementation of a Tensorflow matrix multiplication layer, based on the TensorFlow operator MatMulOp https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/core/kernels/matmul_op.cc.

## Dependencies

### Eigen3

Eigen3 is shipped with the project. If the build script build.sh is used, this version is used by default. Usage of a local installation of Eigen3 is possible, and is the default option used in the CMakeList.txt files when building the modules individually without usage of the build script.

### TensorFlow

The required TensorFlow version is 1.13.1. 

### Tensorpack

The required TensorPack version is 0.9.4.

### Other dependencies for building custom TensorFlow operators

The directory third_party/tensorflow_op_build_dependencies contains a number of header files required to build TensorFlow operators for version 1.13. Most of the headers are from a TensorFlow install build from source, with addition of Abseil headers. This is a kind of messy solution that was only chosen because of the limited time constraints of the master thesis. Future development should address this, and switch to using headers from locally installed TensorFlow and Abseil installations build from source.

## Compilation

To build the submodules of the project, simply run the build script build.sh. The compiler used during main development was GCC 8.1. Because use of the \_\_restrict\_\_ directive is made, the code should also be able to be compiled using clang, but no testing has been done to confirm this.
After the submodules have been successfully build, you can run the mpu_simulator sanity check mpusim_test found in the directory bin/build_mpu_simulator_release, to check if the tool works as intended.

## Usage
