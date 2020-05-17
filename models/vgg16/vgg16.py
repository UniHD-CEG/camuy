#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: vgg16.py

import argparse
import os
import sys
import tensorflow as tf

from tensorpack import *
from tensorpack.tfutils import argscope
from tensorpack.tfutils.summary import *
from tensorpack.utils.gpu import get_num_gpu

sys.path.append(os.path.abspath(""))
sys.path.append(os.path.abspath(""))
sys.path.append(os.path.abspath(""))

from MpuSimConv2D_gradient import *
from MpuSimConv2D import *

from MpuSimMatMul_gradient import *
from MpuSimFullyConnected import *

from imagenet_utils import ImageNetModel, fbresnet_augmentor, get_imagenet_dataflow

session_conf = tf.ConfigProto(
      intra_op_parallelism_threads=1,
      inter_op_parallelism_threads=1)
sess = tf.Session(config=session_conf)


def convnormrelu(x, name, chan):
    x = MpuSimConv2D(name, x, chan, 3)
    x = BatchNorm(name + '_bn', x)
    x = tf.nn.relu(x, name=name + '_relu')
    return x


class Model(ImageNetModel):
    
    def __init__(self,
                data_format='NHWC',
                wd=5e-4,
                learning_rate=0.1,
                activations_datatype_size_byte=1,
                weights_datatype_size_byte=1,
                results_datatype_size_byte=4,
                systolic_array_height=256,
                systolic_array_width=256,
                accumulator_array_height=4096):
        super(Model, self).__init__(data_format, wd)

        self.activations_datatype_size_byte=activations_datatype_size_byte
        self.weights_datatype_size_byte=weights_datatype_size_byte
        self.results_datatype_size_byte=results_datatype_size_byte
        self.systolic_array_height=systolic_array_height
        self.systolic_array_width=systolic_array_width
        self.accumulator_array_height=accumulator_array_height

    def get_logits(self, image):
        constant_init = tf.constant_initializer(1)
        with argscope(MpuSimConv2D, kernel_initializer=constant_init), \
                argscope([MpuSimConv2D, MaxPooling, BatchNorm], data_format='NHWC'), \
                argscope([MpuSimConv2D, MpuSimFullyConnected],
                                            kernel_initializer=constant_init,
                                            activations_datatype_size_byte=self.activations_datatype_size_byte, 
                                            weights_datatype_size_byte=self.weights_datatype_size_byte,
                                            results_datatype_size_byte=self.results_datatype_size_byte,
                                            systolic_array_height=self.systolic_array_height,
                                            systolic_array_width=self.systolic_array_width,
                                            activation_fifo_depth=8,
                                            accumulator_array_height=self.accumulator_array_height,
                                            log_file_output_dir=("/home/kstehle/masters_thesis/"
                                                                    "tensorpack_models_mpusim/vgg16/"
                                                                    "mpu_log/width_height_sweep_constant_pe_count/"),
                                            model_name='vgg16_sys_arr_h_{}_sys_arr_w_{}_acc_arr_h_{}'.format(self.systolic_array_height,
                                                                                                                self.systolic_array_width, 
                                                                                                                self.accumulator_array_height)):
            logits = (LinearWrap(image)
                      .apply(convnormrelu, 'conv1_1', 64)
                      .apply(convnormrelu, 'conv1_2', 64)
                      .MaxPooling('pool1', 2)
                      # 112
                      .apply(convnormrelu, 'conv2_1', 128)
                      .apply(convnormrelu, 'conv2_2', 128)
                      .MaxPooling('pool2', 2)
                      # 56
                      .apply(convnormrelu, 'conv3_1', 256)
                      .apply(convnormrelu, 'conv3_2', 256)
                      .apply(convnormrelu, 'conv3_3', 256)
                      .MaxPooling('pool3', 2)
                      # 28
                      .apply(convnormrelu, 'conv4_1', 512)
                      .apply(convnormrelu, 'conv4_2', 512)
                      .apply(convnormrelu, 'conv4_3', 512)
                      .MaxPooling('pool4', 2)
                      # 14
                      .apply(convnormrelu, 'conv5_1', 512)
                      .apply(convnormrelu, 'conv5_2', 512)
                      .apply(convnormrelu, 'conv5_3', 512)
                      .MaxPooling('pool5', 2)
                      ## 7
                      .MpuSimFullyConnected('fc6', 4096)
                      .tf.nn.relu(name='fc6_relu')
                      .MpuSimFullyConnected('fc7', 4096)
                      .tf.nn.relu(name='fc7_relu')
                      .MpuSimFullyConnected('fc8', 1000)())
        add_param_summary(('.*', ['histogram', 'rms']))
        return logits


def get_config(activations_datatype_size_byte,
                weights_datatype_size_byte,
                results_datatype_size_byte,
                systolic_array_height,
                systolic_array_width,
                accumulator_array_height):
    
    nr_tower = max(get_num_gpu(), 1)
    BASE_LR = 0.01 * (1. / 256.)

    logger.info("Running on {} towers. Batch size per tower: {}".format(nr_tower, 1))

    infs = [ClassificationError('wrong-top1', 'val-error-top1'),
            ClassificationError('wrong-top5', 'val-error-top5')]

    data = QueueInput(FakeData(
            [[1, 224, 224, 3], [1]], 1000, random=False, dtype='uint8'))
    callbacks = []

    return TrainConfig(
                    model=Model(
                            activations_datatype_size_byte=activations_datatype_size_byte,
                            weights_datatype_size_byte=weights_datatype_size_byte,
                            results_datatype_size_byte=results_datatype_size_byte,
                            systolic_array_height=systolic_array_height,
                            systolic_array_width=systolic_array_width,
                            accumulator_array_height=accumulator_array_height),
                    data=data,
                    callbacks=callbacks,
                    steps_per_epoch=1,
                    max_epoch=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--activations-datatype-size-byte',
                            help='activations datatype size in byte',
                            type=int, default=1)
    parser.add_argument('--weights-datatype-size-byte',
                            help='weights datatype size in byte',
                            type=int, default=1)
    parser.add_argument('--results-datatype-size-byte',
                            help='results datatype size in byte',
                            type=int, default=4)
    parser.add_argument('--systolic-array-height',
                            help='systolic array height',
                            type=int, default=256)
    parser.add_argument('--systolic-array-width',
                            help='systolic array width',
                            type=int, default=256)
    parser.add_argument('--accumulator-array-height',
                            help='accumulator array height',
                            type=int, default=4096)
    parser.add_argument('--logdir-id', help='identify of logdir',
                            type=str, default='')
    args = parser.parse_args()

    logger.set_logger_dir(os.path.join('train_log', 'vgg16' + args.logdir_id))

    config = get_config(args.activations_datatype_size_byte,
                                args.weights_datatype_size_byte,
                                args.results_datatype_size_byte,
                                args.systolic_array_height,
                                args.systolic_array_width,
                                args.accumulator_array_height)
    
    launch_train_with_config(config, SimpleTrainer()) 
