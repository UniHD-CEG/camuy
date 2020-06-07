# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow.compat.v1 as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import candidate_sampling_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup
from tensorflow.python.util.tf_export import tf_export

mpu_sim_conv2d_lib = tf.load_op_library('../../bin/build_mpusim_conv2d_release/mpusim-conv2d.so')

def mpusim_separable_conv2d_impl(input,
                                    depthwise_filter,
                                    pointwise_filter,
                                    strides,
                                    padding,
                                    rate=None,
                                    name=None,
                                    data_format=None,
                                    activations_datatype_size_byte=1,
                                    weights_datatype_size_byte=1,
                                    results_datatype_size_byte=4,
                                    systolic_array_height=256,
                                    systolic_array_width=256,
                                    activation_fifo_depth=8,
                                    accumulator_array_height=4096,
                                    log_file_output_dir='.',
                                    model_name='unnamed'):
    
    with ops.name_scope(name, "mpusim_separable_conv2d_impl",
                        [input, depthwise_filter, pointwise_filter]) as name:
        
        input = ops.convert_to_tensor(input, name="tensor_in")
        
        depthwise_filter = ops.convert_to_tensor(depthwise_filter,
                                                    name="depthwise_filter")
        
        pointwise_filter = ops.convert_to_tensor(pointwise_filter,
                                                    name="pointwise_filter")

        depthwise_filter_shape = depthwise_filter.get_shape().with_rank(4)
        
        channels = depthwise_filter_shape.dims[3]

        pointwise_filter_shape = pointwise_filter.get_shape().with_rank(4)
        pointwise_filter_shape.dims[0].assert_is_compatible_with(1)
        pointwise_filter_shape.dims[1].assert_is_compatible_with(1)

        if rate is None:
            rate = [1, 1]

        # The layout of the ops in the graph are expected to be as follows:
        # depthwise_conv2d  // Conv2D op corresponding to deptwise convolution
        # separable_conv2d  // Conv2D op corresponding to the pointwise convolution

        def op(input_converted, _, padding):
                                                    
            inputs = tf.split(input_converted, channels, 3)
            kernels = tf.split(depthwise_filter, channels, 3)
            outputs = [mpu_sim_conv2d_lib.mpu_sim_conv2d(input_block,
                                                            kernel_block,
                                                            activations_datatype_size_byte,
                                                            weights_datatype_size_byte,
                                                            results_datatype_size_byte,
                                                            systolic_array_height,
                                                            systolic_array_width,
                                                            activation_fifo_depth,
                                                            accumulator_array_height,
                                                            log_file_output_dir,
                                                            model_name,
                                                            strides=strides,
                                                            padding=padding)
                        for input_block, kernel_block in zip(inputs, kernels)]
            
            print('Executed depthwise convolution')
            
            return tf.concat(outputs, 3)

        depthwise = nn_ops.with_space_to_batch(input=input,
                                                filter_shape=array_ops.shape(depthwise_filter),
                                                dilation_rate=rate,
                                                padding=padding,
                                                data_format=data_format,
                                                op=op)

        return mpu_sim_conv2d_lib.mpu_sim_conv2d(depthwise,
                                                    pointwise_filter,
                                                    activations_datatype_size_byte,
                                                    weights_datatype_size_byte,
                                                    results_datatype_size_byte,
                                                    systolic_array_height,
                                                    systolic_array_width,
                                                    activation_fifo_depth,
                                                    accumulator_array_height,
                                                    log_file_output_dir,
                                                    model_name,
                                                    strides=[1, 1, 1, 1],
                                                    padding="VALID",
                                                    data_format=data_format,
                                                    name=name)
