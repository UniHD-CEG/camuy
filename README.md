# mpusim

This project provides emulation of computation of convolutional and fully connected layers on weight stationary systolic arrays. It was developed as part of the master thesis "Efficient Design and Mapping of Deep Neural Networks onto Fixed Processing Units". The emulated architecture, based on the Google TPU v1, was named the Matrix Processing Unit, or MPU in short.

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

## Modules

The project consists of the following four modules:

### mpu_simulator

The actual library emulating matrix multiplication computation on weight stationary systolic arrays.
The usage of the library itself without the wrapper and the TensorFlow/Tensorpack operators is described in the following. Beside the described functions, the library also allows for direct communication with the unified buffer, without the provided weight/activation/result matrix memory management.

#### Instantiate a MatrixProcessingUnit object with the desired weight, activation, and accumulator/result data type, systolic array size, activation FIFO depth, accumulator array height, and maximum unified buffer size

The accumulator array height should generally exceed the systolic array height.
TODO: Add minimum activation FIFO depth

Example:

```cpp
MatrixProcessingUnit<WeightDatatype,
                        ActivationDatatype,
                        AccumulatorDatatype> matrixProcessingUnit(systolicArrayWidth,
                                                                    systolicArrayHeight,
                                                                    activationFifoDepth,
                                                                    accumulatorArrayHeight,
                                                                    unifiedBufferSizeByte);
```

#### Set the desired debug output verbosity

The tool implements two debug output verbosity levels. Debug messages are activated by passing `true` to the method `setDebugFlag()`. Verbose debug output can be activated by passing `true` to the method `setDebugOutputVerboseFlag()`.

#### Optional: Select the unified buffer resize mode

The project provides two unified buffer resize modes. In the default dynamic resize mode, the array used for emulation of the unified buffer is resized according to the currently needed space for storing all weight matrices as well as the activation and the result matrix. The memory is extended up to the chosen maximum unified buffer size. If the required memory exceeds this limit, and exception is thrown.
While this approach is memory efficient, the frequent resizing of the unified buffer array incurs some overhead. As an alternative, the project provides the static unified buffer size mode. The emulated MPU can be set to this mode by passing `false` to the method `setUnifiedBufferDynamicResize()`. In this mode, the unified buffer array is allocated on MatrixProcessingUnit object initialization with the given maximum unified buffer size, and only the pointers to the weight/activation/result segments are shifted. When a store operation results in memory usage beyond the allocated size, an exception is thrown.

#### Optional: Register a callback to collect execution metrics

Using the method `registerLogEntryAvailableCallback()`, a callback function can be registered that will be called upon successful matrix multiplication computation. This callback contains execution metrics such as total required iterations and data movements between the functional units and the unified buffer of the emulated MPU.
The `MpuStatisticsLogger` class contained in the mpu_simulator project provides such a callback function that simply stores the received metrics in a CSV file, with a filename generated from the combined directory and name prefix and a weight/activation/accumulator datatype size combination.
Example:

```cpp
MpuStatisticsLogger mpuStatisticsLogger("test", sizeof(WeightDatatype),
                                            sizeof(ActivationDatatype),
                                            sizeof(AccumulatorDatatype));

matrixProcessingUnit.registerLogEntryAvailableCallback(
                        [&mpuStatisticsLogger](MpuStatisticsLogEntry&& mpuStatisticsLogEntry){
    mpuStatisticsLogger.addMpuStatisticsLogEntry(std::move(mpuStatisticsLogEntry));
```

#### Store the weight/activation matrices to the emulated MPU unified buffer

Weight and activation matrices can be stored to the unified buffer in any order. While multiply weight matrices can reside in the emulated MPU unified buffer, only one activation matrix can be stored at any given time.
Weight matrices are stored to the emulated unified buffer using the method `storeWeightMatrix()`. The function requires a string identifier, which will be used during multiplication to identify the the weight matrix to be used. The other parameters are a pointer to the matrix to be copied, and the height and width of the matrix. Input matrices are required to be of the same data type as the choses weight datatype used by the emulated MPU.
Example:

```cpp
matrixProcessingUnit.storeWeightMatrix(weightMatrixNameString,
                                                weightMatrixPtr,
                                                height, width);
```
The activation matrices are stored using the method `storeActivationMatrix()`. The method requires a pointer to the activation matrix to be copied, which has to be of the same datatype as the activation datatype used by the emulated MPU. The other two parameters are the height and width of the input activation matrix.
Example:

```cpp
matrixProcessingUnit.storeActivationMatrix(actMatrixPtr,
                                            height, width);
```
        
#### Run the matrix multiplication

To multiplication process is started using the method `runMultiplication()`. This method requires the name of the weight matrix to be used in the multiplication as a parameter.
Note: The MPU architecture in its current state can only process matrices where !((N > systolicArrayWidth) && (K <= systolicArrayHeight)).

#### Load the result matrix from the emulated MPU unified buffer

After the matrix multiplication process has successfully finished, the result matrix can be retrieved from the emulated unified buffer using the method `loadResultMatrix()`. This method requires a pointer of the same datatype as used for the accumulator/result datatype of the emulated MPU, aswell as the number of elements to be loaded from the result matrix.
Example:

```cpp
matrixProcessingUnit.loadResultMatrix(resultMatrixPtr,
                                        resultMatrixSize);
```

#### Optional: Reset iteration counts/execution metrics

The iteration counts can be reset through the method `resetIterationCounts()`. The other execution metrics can be reset through the method `resetDataMovementAndFootprintMetrics()`.

#### Optional: Reset Memory Management Unit

The Memory Management Unit can be reset by the method `resetMemoryManagementUnit()`. In dynamic allocation mode, the array emulating the unified buffer is freed, and the dope vectors of the stored matrices are deleted. In static allocation mode, this simply results in deletion of the corresponding dope vectors.

### mpusim_wrapper

This library serves as a wrapper for the mpu_simulator library. It ensures that only a single instance of the model is active at any given point. It also ensures that the input matrices are padded for input sizes that are outside the limitations of the emulator. After every multiplication, the iteration count and execution metrics are reset.

### mpusim_conv2d

This project contains the C++ implementation of a TensorFlow conv2d layer, based on the original TensorFlow Conv2DUsingGemmOp operator found in https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/core/kernels/conv_ops_using_gemm.cc, that uses the mpu_simulator through the mpusim_wrapper library for GEMM-based convolution computation. It also provides a Tensorpack 2d convolution operator based on the on the original Tensorpack Conv2d operator found in https://github.com/tensorpack/tensorpack/blob/master/tensorpack/models/conv2d.py, which calls the custom TensorFlow conv2d operator in the background.

### mpusim_fc

This project implements the C++ implementation of a Tensorflow matrix multiplication layer, based on the TensorFlow operator MatMulOp https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/core/kernels/matmul_op.cc.

## Usage of the Tensorpack operators

The custom Tensorpack operators have some additional parameters compared to the standard Tensorpack operators which are used to control the MPU parameters and logging of execution metrics. These are described in the following.

| Parameter                         | Description                                       | Options               |
| --------------------------------- | ------------------------------------------------- | --------------------- |
| `activations_datatype_size_byte`  | Activation datatype size in byte                  | 8, 16, 32, 64         |
| `weights_datatype_size_byte`      | Weight datatype size in byte                      | 8, 16, 32, 64         |
| `results_datatype_size_byte`      | Accumulator/result datatype size in byte          | 8, 16, 32, 64         |
| `systolic_array_height`           | Systolic array height                             | int >= 2              |
| `systolic_array_width`            | Systolic array width                              | int >= 2              |
| `activation_fifo_depth`           | Activation FIFO depth                             | int >= 4              |
| `accumulator_array_height`        | Accumulator array height                          | int >= 4              |
| `log_file_output_dir`             | Directory to which the log file will be written   | Any valid directory   |
| `model_name`                      | Name of the current model                         | Any valid filename    |

