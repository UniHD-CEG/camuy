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

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import candidate_sampling_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_array_ops  # pylint: disable=unused-import
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
    
    """2-D convolution with separable filters.
    Performs a depthwise convolution that acts separately on channels followed by
    a pointwise convolution that mixes channels.  Note that this is separability
    between dimensions `[1, 2]` and `3`, not spatial separability between
    dimensions `1` and `2`.
    In detail,
        output[b, i, j, k] = sum_{di, dj, q, r}
            input[b, strides[1] * i + di, strides[2] * j + dj, q] *
            depthwise_filter[di, dj, q, r] *
            pointwise_filter[0, 0, q * channel_multiplier + r, k]
    `strides` controls the strides for the depthwise convolution only, since
    the pointwise convolution has implicit strides of `[1, 1, 1, 1]`.  Must have
    `strides[0] = strides[3] = 1`.  For the most common case of the same
    horizontal and vertical strides, `strides = [1, stride, stride, 1]`.
    If any value in `rate` is greater than 1, we perform atrous depthwise
    convolution, in which case all values in the `strides` tensor must be equal
    to 1.
    Args:
    input: 4-D `Tensor` with shape according to `data_format`.
    depthwise_filter: 4-D `Tensor` with shape
        `[filter_height, filter_width, in_channels, channel_multiplier]`.
        Contains `in_channels` convolutional filters of depth 1.
    pointwise_filter: 4-D `Tensor` with shape
        `[1, 1, channel_multiplier * in_channels, out_channels]`.  Pointwise
        filter to mix channels after `depthwise_filter` has convolved spatially.
    strides: 1-D of size 4.  The strides for the depthwise convolution for
        each dimension of `input`.
    padding: A string, either `'VALID'` or `'SAME'`.  The padding algorithm.
        See the "returns" section of `tf.nn.convolution` for details.
    rate: 1-D of size 2. The dilation rate in which we sample input values
        across the `height` and `width` dimensions in atrous convolution. If it is
        greater than 1, then all values of strides must be 1.
    name: A name for this operation (optional).
    data_format: The data format for input. Either "NHWC" (default) or "NCHW".
    Returns:
    A 4-D `Tensor` with shape according to 'data_format'. For
        example, with data_format="NHWC", shape is [batch, out_height,
        out_width, out_channels].
    """
    with ops.name_scope(name, "mpusim_separable_conv2d_impl",
                        [input, depthwise_filter, pointwise_filter]) as name:
        
        input = ops.convert_to_tensor(input, name="tensor_in")
        
        depthwise_filter = ops.convert_to_tensor(depthwise_filter,
                                                    name="depthwise_filter")
        
        pointwise_filter = ops.convert_to_tensor(pointwise_filter,
                                                    name="pointwise_filter")

        depthwise_filter_shape = pointwise_filter.get_shape().with_rank(4)
        
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
            return tf.concat(outputs, channel_axis)

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
