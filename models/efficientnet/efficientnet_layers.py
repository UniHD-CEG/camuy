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

import utils

sys.path.append('../..')

from mpusim_conv2d.mpusim_conv2d import *
from mpusim_separable_conv2d.mpusim_depthwise_conv2d_tensorpack import *
from mpusim_fc.mpusim_fully_connected import *

slim = contrib_slim

GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate', 'data_format',
    'num_classes', 'width_coefficient', 'depth_coefficient', 'depth_divisor',
    'min_depth', 'survival_prob', 'relu_fn', 'batch_norm', 'use_se',
    'se_coefficient', 'local_pooling', 'condconv_num_experts',
    'clip_projection_output', 'blocks_args', 'fix_head_stem',
])
# Note: the default value of None is not necessarily valid. It is valid to leave
# width_coefficient, depth_coefficient at None, which is treated as 1.0 (and
# which also allows depth_divisor and min_depth to be left at None).
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)


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

def round_filters(filters, global_params):
    
    """Round number of filters based on depth multiplier."""
    
    orig_f = filters
    multiplier = 1.0
    divisor = 8
    min_depth = None
    
    if not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    
    # Make sure that round down does not go down by more than 10%.
    
    if new_filters < 0.9 * filters:
        new_filters += divisor
        
    logging.info('round_filter input=%s output=%s', orig_f, new_filters)
    
    return int(new_filters)


def round_repeats(repeats, global_params):
    
    """Round number of filters based on depth multiplier."""
    
    multiplier = 1.0
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))



def mb_conv():
    
    block_args = block_args
    batch_norm_momentum = 0.99
    batch_norm_epsilon = 1e-3
    batch_norm = utils.BatchNormalization
    data_format = 'channel_last'
    se_coefficient = 0.25
    
    if data_format == 'channels_first':
        raise ValueError('mpusim convolutional layers are only compatible with NHWC format')
    
    else:
        channel_axis = -1
        spatial_dims = [1, 2]

    relu_fn = global_params.relu_fn or swish

    clip_projection_output = global_params.clip_projection_output

    conv_cls = mpusim_conv2d
    depthwise_conv_cls = utils.DepthwiseConv2D
    
    """Builds block according to the arguments."""

    filters = block_args.input_filters * block_args.expand_ratio
    kernel_size = block_args.kernel_size

    # Fused expansion phase. Called if using fused convolutions
    
    fused_conv = conv_cls(filters=filters,
                            kernel_size=[kernel_size, kernel_size],
                            strides=block_args.strides,
                            kernel_initializer=conv_kernel_initializer,
                            padding='same',
                            data_format=data_format,
                            use_bias=False)

    # Expansion phase. Called if not using fused convolutions and expansion
    # phase is necessary.
    
    expand_conv = conv_cls(filters=filters,
                                    kernel_size=[1, 1],
                                    strides=[1, 1],
                                    kernel_initializer=conv_kernel_initializer,
                                    padding='same',
                                    data_format=data_format,
                                    use_bias=False)
    
    bn0 = batch_norm(axis=channel_axis,
                        momentum=batch_norm_momentum,
                        epsilon=batch_norm_epsilon)

    # Depth-wise convolution phase. Called if not using fused convolutions.

    depthwise_conv = depthwise_conv_cls(kernel_size=[kernel_size, kernel_size],
                                                strides=block_args.strides,
                                                depthwise_initializer=conv_kernel_initializer,
                                                padding='same',
                                                data_format=data_format,
                                                use_bias=False)

    bn1 = batch_norm(axis=channel_axis,
                        momentum=batch_norm_momentum,
                        epsilon=batch_norm_epsilon)

    if has_se:
        num_reduced_filters = max(1, int(block_args.input_filters* \
                                                (block_args.se_ratio* \
                                                (se_coefficient if se_coefficient else 1))))

    # Output phase
    
    filters = block_args.output_filters
    
    project_conv = conv_cls(filters=filters,
                                        kernel_size=[1, 1],
                                        strides=[1, 1],
                                        kernel_initializer=conv_kernel_initializer,
                                        padding='same',
                                        data_format=data_format,
                                        use_bias=False)
    
    bn2 = batch_norm(axis=channel_axis,
                        momentum=batch_norm_momentum,
                        epsilon=batch_norm_epsilon)
    
    x = inputs

    fused_conv_fn = fused_conv
    expand_conv_fn = expand_conv
    depthwise_conv_fn = depthwise_conv
    project_conv_fn = project_conv

    if block_args.fused_conv:
        
        # If use fused mbconv, skip expansion and use regular conv
        x = relu_fn(bn1(fused_conv_fn(x), training=training))
        
    else:
        # Otherwise, first apply expansion and then apply depthwise conv
        if block_args.expand_ratio != 1:
                x = relu_fn(bn0(expand_conv_fn(x), training=training))

    x = relu_fn(bn1(mpusim_depthwise_convolution2d(x,
                                                    kernel_size=[kernel_size, kernel_size],
                                                    strides=block_args.strides,
                                                    depthwise_initializer=conv_kernel_initializer,
                                                    padding='same',
                                                    data_format=data_format,
                                                    use_bias=False))

    with tf.variable_scope('se'):
        
        x = tf.reduce_mean(x,
                            spatial_dims,
                            keepdims=True)
        
        x = slim.conv2d(x,
                        num_reduced_filters,
                        kernel_size=[1, 1],
                        strides=[1, 1],
                        kernel_initializer=conv_kernel_initializer,
                        padding='same',
                        data_format=data_format,
                        use_bias=True)
            
        x = relu_fn(x)
            
        x = slim.conv2d(x,
                        filters,
                        kernel_size=[1, 1],
                        strides=[1, 1],
                        kernel_initializer=conv_kernel_initializer,
                        padding='same',
                        data_format=data_format,
                        use_bias=True)
        
        x = tf.sigmoid(se_tensor)*input_tensor

    x = bn2(project_conv_fn(x), training=training)
    
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

