# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
# Modifications copyright (c) 2020 Computing Systems Group.
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
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops

from tensorpack.compat import tfv1 as tf

from tensorpack.tfutils.common import get_tf_version_tuple
from tensorpack.utils.argtools import get_data_format, shape2d, shape4d, log_once
from tensorpack.models.common import VariableHolder, layer_register
from tensorpack.models.tflayer import convert_to_tflayer_args, rename_get_variable

from .mpusim_depthwise_conv2d import mpusim_depthwise_conv2d

__all__ = ['mpusim_depthwise_convolution2d']

@layer_register(log_shape=True)
def mpusim_depthwise_convolution2d(inputs,
                                    kernel_size,
                                    strides=(1, 1),
                                    padding='valid',
                                    depth_multiplier=1,
                                    data_format='channels_last',
                                    activation=None,
                                    use_bias=True,
                                    depthwise_initializer='glorot_uniform',
                                    bias_initializer='zeros',
                                    depthwise_regularizer=None,
                                    bias_regularizer=None,
                                    depthwise_constraint=None,
                                    bias_constraint=None,
                                    activations_datatype_size_byte=1,
                                    weights_datatype_size_byte=1,
                                    results_datatype_size_byte=4,
                                    systolic_array_height=256,
                                    systolic_array_width=256,
                                    activation_fifo_depth=8,
                                    accumulator_array_height=4096,
                                    log_file_output_dir='.',
                                    model_name='unnamed'):
        
    #depthwise_initializer = initializers.get(depthwise_initializer)
    #depthwise_regularizer = regularizers.get(depthwise_regularizer)
    #depthwise_constraint = constraints.get(depthwise_constraint)
    #bias_initializer = initializers.get(bias_initializer)

    data_format = get_data_format(data_format, keras_mode=False)
    input_shape = inputs.get_shape().as_list()
    
    strides = shape4d(strides, data_format=data_format)

    if len(input_shape) < 4:
        raise ValueError('Inputs to `mpusim_depthwise_conv2d` should have rank 4. '
                                        'Received input shape:', str(input_shape))
    
    if data_format == 'NCHW':
        raise ValueError('mpusim_depthwise_convolution2d '
                            'only supports NHWC data format')
    else:
        channel_axis = 3
        
    if input_shape[channel_axis] is None:
        raise ValueError('The channel dimension of the inputs to '
                                '`mpusim_depthwise_convolution2d` '
                                'should be defined. Found `None`.')
    
    input_dim = int(input_shape[channel_axis])
    
    depthwise_kernel_shape = (kernel_size[0],
                                kernel_size[1],
                                input_dim,
                                depth_multiplier)
        
    depthwise_kernel = tf.get_variable('W', shape=depthwise_kernel_shape,
                                        initializer=depthwise_initializer,
                                        regularizer=depthwise_regularizer,
                                        constraint=depthwise_constraint)

    if use_bias:
        biases = tf.get_variable('b', shape=(input_dim*depth_multiplier,),
                                                initializer=bias_initializer,
                                                regularizer=bias_regularizer,
                                                constraint=bias_constraint)

    result = mpusim_depthwise_conv2d(inputs,
                                        depthwise_kernel,
                                        strides=strides,
                                        padding=padding,
                                        data_format=data_format,
                                        activations_datatype_size_byte=activations_datatype_size_byte,
                                        weights_datatype_size_byte=weights_datatype_size_byte,
                                        results_datatype_size_byte=results_datatype_size_byte,
                                        systolic_array_height=systolic_array_height,
                                        systolic_array_width=systolic_array_width,
                                        activation_fifo_depth=activation_fifo_depth,
                                        accumulator_array_height=accumulator_array_height,
                                        log_file_output_dir=log_file_output_dir,
                                        model_name=model_name)

    if use_bias:
        result = tf.nn.bias_add(result,
                                    bias,
                                    data_format=data_format)

    if activation is not None:
        result = activation(result)
        
    result = tf.identity(result, name='output')

    result.variables = VariableHolder(W=depthwise_kernel)
    
    if use_bias:
        result.variables.b=biases
        
    return result

