#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: googlenet_model.py

import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorpack.models import *
from tensorpack.tfutils.argscope import argscope, get_arg_scope

from mpusim_conv2d.mpusim_conv2d import *
from mpusim_fc.mpusim_fully_connected import *

def inception_block(l, name, ch_1x1, ch_3x3, ch_5x5, is_last_block=False, is_last=False):
    data_format = 'NHWC'
    with tf.variable_scope(name):
        conv1x1 = mpusim_conv2d('1x1', l, ch_1x1, 1, activation=BNReLU if not is_last_block else tf.identity)
        conv3x3_reduce = mpusim_conv2d('3x3_reduce', l, ch_3x3, 1, activation=BNReLU)
        conv3x3 = mpusim_conv2d('3x3', conv3x3_reduce, ch_3x3, 3, activation=BNReLU if not is_last_block else tf.identity)
        conv5x5_reduce = mpusim_conv2d('5x5_reduce', l, ch_5x5, 1, activation=BNReLU)
        conv5x5 = mpusim_conv2d('5x5', conv5x5_reduce, ch_5x5, 5, activation=BNReLU if not is_last_block else tf.identity)
        if is_last_block and not is_last:
            conv1x1 = MaxPooling('pool_1x1', conv1x1, shape=3, stride=2, padding='SAME')
            conv1x1 = BNReLU('conv1x1_bn', conv1x1)
            conv3x3 = MaxPooling('pool_3x3', conv3x3, shape=3, stride=2, padding='SAME')
            conv3x3 = BNReLU('conv3x3_bn', conv3x3)
            conv5x5 = MaxPooling('pool_5x5', conv5x5, shape=3, stride=2, padding='SAME')
            conv5x5 = BNReLU('conv5x5_bn', conv5x5)
        l = tf.concat([
            conv1x1,
            conv3x3,
            conv5x5], 3, name='concat')
        if is_last:
            l = BNReLU('output_bn', l)
        return l


def googlenet_backbone(image,
                        activations_datatype_size_byte,
                        weights_datatype_size_byte,
                        results_datatype_size_byte,
                        systolic_array_height,
                        systolic_array_width,
                        accumulator_array_height,
                        mpusim_logdir):

    constant_init = tf.constant_initializer(1)
    with argscope(mpusim_conv2d,
                    data_format='NHWC',
                    use_bias=False), \
            argscope([mpusim_conv2d, mpusim_fully_connected],
                        nl=tf.identity,
                        kernel_initializer=constant_init, 
                        activations_datatype_size_byte=activations_datatype_size_byte, 
                        weights_datatype_size_byte=weights_datatype_size_byte,
                        results_datatype_size_byte=results_datatype_size_byte,
                        systolic_array_height=systolic_array_height,
                        systolic_array_width=systolic_array_width,
                        activation_fifo_depth=8,
                        accumulator_array_height=accumulator_array_height,
                        log_file_output_dir=mpusim_logdir,
                        model_name='googlenet_sys_arr_h_{}_sys_arr_w_{}_acc_arr_h_{}'.format(systolic_array_height,
                                                                                                systolic_array_width, 
                                                                                                accumulator_array_height)):
        l = mpusim_conv2d('conv1', image, 64, 7, stride=2)
        l = MaxPooling('pool1', l, shape=3, stride=2, padding='SAME')
        l = BNReLU('pool1/out', l)
        l = mpusim_conv2d('conv2/3x3_reduce', l, 192, 1, activation=BNReLU)
        l = mpusim_conv2d('conv2/3x3', l, 192, 3)
        l = MaxPooling('pool2', l, shape=3, stride=2, padding='SAME')
        l = BNReLU('pool2/out', l)
        l = inception_block(l, 'inception_3a', 96, 128, 32)
        l = inception_block(l, 'inception_3b', 192, 192, 96, is_last_block=True)
        l = inception_block(l, 'inception_4a', 256, 208, 48)
        l = inception_block(l, 'inception_4b', 224, 224, 64)
        l = inception_block(l, 'inception_4c', 192, 256, 64)
        l = inception_block(l, 'inception_4d', 176, 288, 64)
        l = inception_block(l, 'inception_4e', 384, 320, 128, is_last_block=True)
        l = inception_block(l, 'inception_5a', 384, 320, 128)
        l = inception_block(l, 'inception_5b', 512, 384, 128, is_last_block=True, is_last=True)
        l = GlobalAvgPooling('pool5',  l)
        return mpusim_fully_connected('linear', l, out_dim=1000, activation=tf.identity)
        
