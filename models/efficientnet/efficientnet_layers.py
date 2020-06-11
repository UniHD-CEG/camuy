# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Contains definitions for EfficientNet model.

[1] Mingxing Tan, Quoc V. Le
  EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
  ICML'19, https://arxiv.org/abs/1905.11946
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import math

import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow.contrib import slim as contrib_slim

from tensorpack import *

import utils

sys.path.append('../..')

from mpusim_conv2d.mpusim_conv2d import *
from mpusim_separable_conv2d.mpusim_depthwise_conv2d_tensorpack import *

slim = contrib_slim

def swish(features, use_native=True, use_hard=False):
    
    """Computes the Swish activation function.

    We provide three alternnatives:
    - Native tf.nn.swish, use less memory during training than composable swish.
    - Quantization friendly hard swish.
    - A composable swish, equivalant to tf.nn.swish, but more general for
        finetuning and TF-Hub.

    Args:
    features: A `Tensor` representing preactivation values.
    use_native: Whether to use the native swish from tf.nn that uses a custom
        gradient to reduce memory usage, or to use customized swish that uses
        default TensorFlow gradient computation.
    use_hard: Whether to use quantization-friendly hard swish.

    Returns:
    The activation value.
    """
    
    if use_native and use_hard:
        raise ValueError('Cannot specify both use_native and use_hard.')

    if use_native:
        return tf.nn.swish(features)

    if use_hard:
        return features * tf.nn.relu6(features + np.float32(3)) * (1. / 6.)

    features = tf.convert_to_tensor(features, name='features')
    return features * tf.nn.sigmoid(features)

def mb_conv(name,
            inputs,
            kernel_size,
            strides,
            expand_ratio,
            input_filters,
            output_filters,
            conv_kernel_initializer=tf.constant_initializer(1)):
    
    batch_norm_momentum = 0.99
    batch_norm_epsilon = 1e-3
    data_format = 'channel_last'
    se_ratio = 0.25
    
    channel_axis = -1
    spatial_dims = [1, 2]
    
    with tf.variable_scope(name):

        with argscope([BatchNorm],
                        axis=channel_axis,
                        momentum=batch_norm_momentum,
                        epsilon=batch_norm_epsilon):
                
            if expand_ratio != 1:
                x = tf.nn.swish(BatchNorm('bn0',
                                            mpusim_conv2d('expand_conv',
                                                            inputs,
                                                            filters=input_filter*expand_ratio,
                                                            kernel_size=[1, 1],
                                                            strides=[1, 1],
                                                            kernel_initializer=conv_kernel_initializer,
                                                            padding='same',
                                                            data_format=data_format,
                                                            use_bias=False)))

            x = tf.nn.swish(BatchNorm('bn1',
                                        mpusim_depthwise_convolution2d('depthwise_conv',
                                                                        x,
                                                                        kernel_size=[kernel_size, kernel_size],
                                                                        strides=block_args.strides,
                                                                        depthwise_initializer=conv_kernel_initializer,
                                                                        padding='same',
                                                                        data_format=data_format,
                                                                        use_bias=False)))

            with tf.variable_scope('se'):
                
                se = tf.reduce_mean(x,
                                    spatial_dims,
                                    keepdims=True)
                
                num_reduced_filters = max(1, int(input_filters*se_ratio))
                
                se = slim.conv2d(se,
                                    num_reduced_filters,
                                    kernel_size=[1, 1],
                                    strides=[1, 1],
                                    kernel_initializer=conv_kernel_initializer,
                                    padding='same',
                                    data_format=data_format,
                                    use_bias=True)
                    
                se = tf.nn.swish(se)
                    
                se = slim.conv2d(se,
                                    input_filter*expand_ratio,
                                    kernel_size=[1, 1],
                                    strides=[1, 1],
                                    kernel_initializer=conv_kernel_initializer,
                                    padding='same',
                                    data_format=data_format,
                                    use_bias=True)
                
                x = tf.sigmoid(se)*x

            x = BatchNorm('bn2',
                            mpusim_conv2d('project_conv',
                                            x,
                                            filters=output_filters,
                                            kernel_size=[1, 1],
                                            strides=[1, 1],
                                            kernel_initializer=conv_kernel_initializer,
                                            padding='same',
                                            data_format=data_format,
                                            use_bias=False))
            
            # Add identity so that quantization-aware training can insert quantization
            # ops correctly
            
            x = tf.identity(x)

            if block_args.id_skip:
                if all(s == 1 for s in block_args.strides) and \
                        inputs.get_shape().as_list()[-1] == x.get_shape().as_list()[-1]:
                    if survival_prob:
                        x = utils.drop_connect(x, training, survival_prob)
                    x = tf.add(x, inputs)
            
            return x

