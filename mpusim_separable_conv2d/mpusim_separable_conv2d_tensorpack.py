# coding=utf-8
# -*- coding: utf-8 -*-
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

import functools
import six
import tensorflow.compat.v1 as tf
from tf_slim.layers import initializers
from tf_slim.layers import utils
from tf_slim.ops import variables
from tf_slim.ops.arg_scope import add_arg_scope

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.layers import base
from tensorflow.python.layers import convolutional as convolutional_layers
from tensorflow.python.layers import core as core_layers
from tensorflow.python.layers import normalization as normalization_layers
from tensorflow.python.layers import pooling as pooling_layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.training import moving_averages

from tensorpack.tfutils.common import get_tf_version_tuple
from tensorpack.utils.argtools import get_data_format, shape2d, shape4d, log_once
from tensorpack.models.common import VariableHolder, layer_register
from tensorpack.models.tflayer import convert_to_tflayer_args, rename_get_variable

from ..mpusim_depthwise_conv2d import mpusim_depthwise_conv2d
from ..mpusim_separable_conv2d import mpusim_separable_conv2d

__all__ = ['mpusim_separable_convolution2d']

@layer_register(log_shape=True)
def mpusim_separable_convolution2d(inputs,
                                    num_outputs,
                                    kernel_size,
                                    depth_multiplier=1,
                                    stride=1,
                                    padding='SAME',
                                    data_format='NHWC',
                                    rate=1,
                                    activation_fn=nn.relu,
                                    normalizer_fn=None,
                                    normalizer_params=None,
                                    weights_initializer=initializers.xavier_initializer(),
                                    pointwise_initializer=None,
                                    weights_regularizer=None,
                                    biases_initializer=init_ops.zeros_initializer(),
                                    biases_regularizer=None,
                                    reuse=None,
                                    variables_collections=None,
                                    outputs_collections=None,
                                    trainable=True,
                                    scope=None,
                                    activations_datatype_size_byte=1,
                                    weights_datatype_size_byte=1,
                                    results_datatype_size_byte=4,
                                    systolic_array_height=256,
                                    systolic_array_width=256,
                                    activation_fifo_depth=8,
                                    accumulator_array_height=4096,
                                    log_file_output_dir='.',
                                    model_name='unnamed'):
   
    if data_format is not 'NHWC':
        raise ValueError('data_format has to be either NCHW or NHWC.')
    
    layer_variable_getter = _build_variable_getter({
        'bias': 'biases',
        'depthwise_kernel': 'depthwise_weights',
        'pointwise_kernel': 'pointwise_weights'
    })

    with variable_scope.variable_scope(scope,
                                        'separable_convolution2d',
                                        [inputs],
                                        reuse=reuse,
                                        custom_getter=layer_variable_getter) as sc:
        inputs = ops.convert_to_tensor(inputs)

        if pointwise_initializer is None:
            pointwise_initializer = weights_initializer

        df = 'channels_last'
    
        if num_outputs is not None:
            
            # Apply separable conv using the mpusim_separable_conv2d layer.
            
            layer = \
                mpusim_separable_conv2d.mpusim_separable_conv2d(filters=num_outputs,
                                                                kernel_size=kernel_size,
                                                                strides=stride,
                                                                padding=padding,
                                                                data_format=df,
                                                                dilation_rate=utils.two_element_tuple(rate),
                                                                activation=None,
                                                                depth_multiplier=depth_multiplier,
                                                                use_bias=not normalizer_fn and biases_initializer,
                                                                depthwise_initializer=weights_initializer,
                                                                pointwise_initializer=pointwise_initializer,
                                                                bias_initializer=biases_initializer,
                                                                depthwise_regularizer=weights_regularizer,
                                                                pointwise_regularizer=weights_regularizer,
                                                                bias_regularizer=biases_regularizer,
                                                                activity_regularizer=None,
                                                                trainable=trainable,
                                                                name=sc.name,
                                                                dtype=inputs.dtype.base_dtype,
                                                                _scope=sc,
                                                                _reuse=reuse,
                                                                activations_datatype_size_byte=activations_datatype_size_byte,
                                                                weights_datatype_size_byte=weights_datatype_size_byte,
                                                                results_datatype_size_byte=results_datatype_size_byte,
                                                                systolic_array_height=systolic_array_height,
                                                                systolic_array_width=systolic_array_width,
                                                                activation_fifo_depth=activation_fifo_depth,
                                                                accumulator_array_height=accumulator_array_height,
                                                                log_file_output_dir=log_file_output_dir,
                                                                model_name=model_name)
            outputs = layer.apply(inputs)

            # Add variables to collections.
            _add_variable_to_collections(layer.depthwise_kernel,
                                        variables_collections, 'weights')
            _add_variable_to_collections(layer.pointwise_kernel,
                                        variables_collections, 'weights')
            
        if layer.bias is not None:
                _add_variable_to_collections(layer.bias,
                                                variables_collections,
                                                'biases')

        if normalizer_fn is not None:
            normalizer_params = normalizer_params or {}
            outputs = normalizer_fn(outputs, **normalizer_params)
            
        else:
            
            # Actually apply depthwise conv instead of separable conv.
            
            dtype = inputs.dtype.base_dtype
            kernel_h, kernel_w = utils.two_element_tuple(kernel_size)
            stride_h, stride_w = utils.two_element_tuple(stride)
            
            num_filters_in = utils.channel_dimension(inputs.get_shape(),
                                                            df, min_rank=4)
            
            weights_collections = utils.get_variable_collections(variables_collections,
                                                                                'weights')

            depthwise_shape = [kernel_h, kernel_w, num_filters_in, depth_multiplier]
            
            depthwise_weights = variables.model_variable('depthwise_weights',
                                                            shape=depthwise_shape,
                                                            dtype=dtype,
                                                            initializer=weights_initializer,
                                                            regularizer=weights_regularizer,
                                                            trainable=trainable,
                                                            collections=weights_collections)
            
            strides = [1, 1, stride_h, stride_w] if data_format.startswith('NC') \
                                                        else [1, stride_h, stride_w, 1]

            outputs = \
                mpusim_depthwise_conv2d.mpusim_depthwise_conv2d(inputs,
                                                                depthwise_weights,
                                                                strides,
                                                                padding,
                                                                rate=utils.two_element_tuple(rate),
                                                                activations_datatype_size_byte=activations_datatype_size_byte,
                                                                weights_datatype_size_byte=weights_datatype_size_byte,
                                                                results_datatype_size_byte=results_datatype_size_byte,
                                                                systolic_array_height=systolic_array_height,
                                                                systolic_array_width=systolic_array_width,
                                                                activation_fifo_depth=activation_fifo_depth,
                                                                accumulator_array_height=accumulator_array_height,
                                                                log_file_output_dir=log_file_output_dir,
                                                                model_name=model_name)
            
            num_outputs = depth_multiplier*num_filters_in

            if normalizer_fn is not None:
                normalizer_params = normalizer_params or {}
                outputs = normalizer_fn(outputs, **normalizer_params)
                
            else:
                if biases_initializer is not None:
                    
                    biases_collections = \
                            utils.get_variable_collections(variables_collections,
                                                                            'biases')
                    
                    biases = variables.model_variable('biases',
                                                        shape=[num_outputs,],
                                                        dtype=dtype,
                                                        initializer=biases_initializer,
                                                        regularizer=biases_regularizer,
                                                        trainable=trainable,
                                                        collections=biases_collections)
                    
                    outputs = nn.bias_add(outputs, biases, data_format=data_format)

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return utils.collect_named_outputs(outputs_collections, sc.name, outputs)
