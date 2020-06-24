# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
# Modifications copyright 2020 Kevin Stehle.
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

import sys
import math
import collections
import functools

import numpy as np
import tensorflow.compat.v1 as tf

from tensorpack import *

sys.path.append('../..')

from mpusim_conv2d.mpusim_conv2d import *
from mpusim_depthwise_conv2d.mpusim_depthwise_convolution2d import *

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
    data_format = 'channels_last'
    se_ratio = 0.25
    
    channel_axis = -1
    spatial_dims = [1, 2]
    
    x = inputs
    
    with tf.variable_scope(name):

        with argscope([BatchNorm],
                        data_format=data_format,
                        momentum=batch_norm_momentum,
                        epsilon=batch_norm_epsilon):
                
            if expand_ratio != 1:
                x = tf.nn.swish(BatchNorm('bn0',
                                            mpusim_conv2d('expand_conv',
                                                            inputs,
                                                            filters=input_filters*expand_ratio,
                                                            kernel_size=[1, 1],
                                                            strides=[1, 1],
                                                            kernel_initializer=conv_kernel_initializer,
                                                            padding='SAME',
                                                            data_format=data_format,
                                                            use_bias=False)))

            x = tf.nn.swish(BatchNorm('bn1',
                                        mpusim_depthwise_convolution2d('depthwise_conv',
                                                                        x,
                                                                        kernel_size=[kernel_size, kernel_size],
                                                                        strides=strides,
                                                                        depthwise_initializer=conv_kernel_initializer,
                                                                        padding='SAME',
                                                                        data_format=data_format,
                                                                        use_bias=False)))

            with tf.variable_scope('se'):
                
                se = tf.reduce_mean(x,
                                    spatial_dims,
                                    keepdims=True)
                
                num_reduced_filters = max(1, int(input_filters*se_ratio))
                
                se = Conv2D('reduce',
                            se,
                            num_reduced_filters,
                            kernel_size=[1, 1],
                            strides=[1, 1],
                            kernel_initializer=conv_kernel_initializer,
                            padding='SAME',
                            data_format=data_format,
                            use_bias=True)
                    
                se = tf.nn.swish(se)
                    
                se = Conv2D('expand',
                            se,
                            input_filters*expand_ratio,
                            kernel_size=[1, 1],
                            strides=[1, 1],
                            kernel_initializer=conv_kernel_initializer,
                            padding='SAME',
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
                                            padding='SAME',
                                            data_format=data_format,
                                            use_bias=False))
            
            # Add identity so that quantization-aware training can insert quantization
            # ops correctly
            
            x = tf.identity(x)
            
            return x

