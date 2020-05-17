#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: resnet_model.py

import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorpack.models import *
from tensorpack.tfutils.argscope import argscope, get_arg_scope

from MpuSimConv2D import *
from MpuSimFullyConnected import *

def resnet_shortcut(l, n_out, stride, activation=tf.identity, block_type='B'):
    data_format = 'NHWC'
    n_in = l.get_shape().as_list()[3]
    
    # Change dimension when channel is not the same
    if n_in != n_out:
        if block_type == 'B':
            return MpuSimConv2D('convshortcut', l, n_out, 1, stride=stride, activation=activation)
        else:
            l = AvgPooling('poolshortcut', l, stride, stride, padding='VALID')
            l = tf.pad(l, [[0, 0], [0, 0], [0, 0], [0, n_out - n_in]], 'CONSTANT')
            return l
    else:
        return l


def apply_preactivation(l, preact, block_func):
    if preact == 'bnrelu':
        shortcut = l  # preserve identity mapping
        l = BNReLU('preact', l)
    elif preact == 'first':
        if block_func == 'basic':
            shortcut = l
        else:
            shortcut = l
    else:
        shortcut = l
    return l, shortcut


def preresnet_basicblock(l, ch_out, stride, preact, block_type='B'):
    l, shortcut = apply_preactivation(l, preact, 'basic')
    l = MpuSimConv2D('conv1', l, ch_out, 3, stride=stride, activation=BNReLU)
    l = MpuSimConv2D('conv2', l, ch_out, 3, activation=BNReLU)
    return l + resnet_shortcut(shortcut, ch_out, stride, activation=BNReLU, block_type=block_type)


def preresnet_bottleneck(l, ch_out, stride, preact, block_type='A'):
    # Stride is applied on the second conv, following fb.resnet.torch
    l, shortcut = apply_preactivation(l, preact, 'basic')
    l = MpuSimConv2D('conv1', l, ch_out, 1, activation=BNReLU)
    l = MpuSimConv2D('conv2', l, ch_out, 3, stride=stride, activation=BNReLU)
    l = MpuSimConv2D('conv3', l, ch_out*4, 1, activation=BNReLU)
    return l + resnet_shortcut(shortcut, ch_out*4, stride, activation=BNReLU, block_type=block_type)


def preresnet_group(l, name, block_func, features, count, stride, is_last=False):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                # first block doesn't need activation
                if i == 0 and stride == 1:
                    preact = 'first'
                elif i == 0:
                    preact = 'no_preact'
                else:
                    preact = 'bnrelu'
                l = block_func(l, features,
                               stride if i == 0 else 1,
                               preact, block_type='B')
        # End of each group need an extra activation
        l = BNReLU('bnlast', l)
    return l


def preresnet_group_typeA(l, name, block_func, features, count, stride, is_last=False):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                # first block doesn't need activation
                if i == 0 and stride == 1:
                    preact = 'first'
                elif i == 0:
                    preact = 'first'
                else:
                    preact = 'bnrelu'
                l = block_func(l, features,
                               stride if i == 0 else 1,
                               preact, block_type='A')
        # end of each group need an extra activation
        l = BNReLU('bnlast', l)
    return l


def resnet_basicblock(l, ch_out, stride):
    shortcut = l
    l = MpuSimConv2D('conv1', l, ch_out, 3, stride=stride, activation=BNReLU)
    l = MpuSimConv2D('conv2', l, ch_out, 3, activation=BNReLU)
    return l + resnet_shortcut(shortcut, ch_out, stride, activation=BNReLU)


def resnet_bottleneck(l, ch_out, stride, stride_first=False):
    """
    stride_first: Original resnet put stride on first conv. fb.resnet.torch put stride on second conv.
    """
    shortcut = l
    l = MpuSimConv2D('conv1', l, ch_out, 1, stride=stride if stride_first else 1, activation=BNReLU)
    l = MpuSimConv2D('conv2', l, ch_out, 3, stride=1 if stride_first else stride, activation=BNReLU)
    l = MpuSimConv2D('conv3', l, ch_out*4, 1, activation=BNReLU)
    return l + resnet_shortcut(shortcut, ch_out*4, stride, activation=BNReLU)


def resnet_group(l, name, block_func, features, count, stride, is_last=False):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                l = block_func(l, features, stride if i == 0 else 1)
                # End of each block need an activation
                l = tf.nn.relu(l)
    return l


def resnet_backbone(image,
                        resnet_depth,
                        num_blocks,
                        group_func,
                        block_func,
                        activations_datatype_size_byte,
                        weights_datatype_size_byte,
                        results_datatype_size_byte,
                        systolic_array_height,
                        systolic_array_width,
                        accumulator_array_height):
    
    constant_init = tf.constant_initializer(1)
    with argscope(MpuSimConv2D, data_format='NHWC'), \
            argscope([MpuSimConv2D, MpuSimFullyConnected],
                                    activation=tf.identity,
                                    use_bias=False,
                                    kernel_initializer=constant_init,
                                    activations_datatype_size_byte=activations_datatype_size_byte, 
                                    weights_datatype_size_byte=weights_datatype_size_byte,
                                    results_datatype_size_byte=results_datatype_size_byte,
                                    systolic_array_height=systolic_array_height,
                                    systolic_array_width=systolic_array_width,
                                    activation_fifo_depth=8,
                                    accumulator_array_height=accumulator_array_height,
                                    log_file_output_dir=("mpu_log/resnet_{}/width_height_sweep".format(resnet_depth)),
                                    model_name='resnet_{}_sys_arr_h_{}_sys_arr_w_{}_acc_arr_h_{}'.format(resnet_depth,
                                                                                                            systolic_array_height,
                                                                                                            systolic_array_width, 
                                                                                                            accumulator_array_height)):

        l = MpuSimConv2D('conv0', image, 64, 7, stride=2, activation=BNReLU)
        l = MaxPooling('pool0', l, shape=3, stride=2, padding='SAME')
        l = group_func(l, 'group0', block_func, 64, num_blocks[0], 1)
        l = group_func(l, 'group1', block_func, 128, num_blocks[1], 2)
        l = group_func(l, 'group2', block_func, 256, num_blocks[2], 2)
        l = group_func(l, 'group3', block_func, 512, num_blocks[3], 2, is_last=True)
        l = GlobalAvgPooling('gap', l)
        return MpuSimFullyConnected('linear', l, 1000)
