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
"""Implementation of Neural Net (NN) functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import math

import tensorflow.compat.v1 as tf
from tensorflow.python.compat import compat
from tensorflow.python.distribute import distribution_strategy_context as ds
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import candidate_sampling_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_array_ops  # pylint: disable=unused-import
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import util as losses_util
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup
from tensorflow.python.util.tf_export import tf_export

from tensorflow.python.keras.layers import subtract as tf_subtract
from tensorflow.python.keras.backend import eval as keras_eval

mpu_sim_conv2d_lib = tf.load_op_library('../../bin/build_mpusim_conv2d_release/mpusim-conv2d.so')

def mpusim_depthwise_conv2d(input,
                                filter,
                                strides,
                                padding,
                                rate=None,
                                name=None,
                                data_format=None,
                                dilations=None,
                                activations_datatype_size_byte=1,
                                weights_datatype_size_byte=1,
                                results_datatype_size_byte=4,
                                systolic_array_height=256,
                                systolic_array_width=256,
                                activation_fifo_depth=8,
                                accumulator_array_height=4096,
                                log_file_output_dir='.',
                                model_name='unnamed'):

    rate = deprecated_argument_lookup("dilations", dilations, "rate", rate)
    
    with ops.name_scope("mpusim_depthwise_conv2d", [input, filter]) as name:
        input = ops.convert_to_tensor(input, name="tensor_in")
        filter = ops.convert_to_tensor(filter, name="filter_in")
        
        if rate is None:
            rate = [1, 1]
            
        channels = input.get_shape().with_rank(4).dims[3]
        
        #print('Depthwise convolution shape: {}'.format(filter.get_shape()))

        def op(input_converted, _, padding):
            
            inputs = tf.split(input_converted, channels, 3)
            kernels = tf.split(filter, channels, 2)
                
            outputs = []
            convolution_count = 0
                
            for input_block, kernel_block in zip(inputs, kernels):
                
                with ops.name_scope("_{}".format(convolution_count)) as name:
                
                    channel_output = mpu_sim_conv2d_lib.mpu_sim_conv2d(input_block,
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
                
                    outputs.append(channel_output)
                
                    convolution_count += 1
            
            return tf.concat(outputs, 3)

        return nn_ops.with_space_to_batch(input=input,
                                            filter_shape=array_ops.shape(filter),
                                            dilation_rate=rate,
                                            padding=padding,
                                            data_format=data_format,
                                            op=op)
