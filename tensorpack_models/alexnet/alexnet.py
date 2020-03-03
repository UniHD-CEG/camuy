#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: alexnet.py

import argparse
import numpy as np
import os
import sys
import cv2
import tensorflow as tf

from tensorpack import *
from tensorpack.dataflow import imgaug
from tensorpack.tfutils import argscope
from tensorpack.utils.gpu import get_num_gpu

sys.path.append(os.path.abspath(""))
sys.path.append(os.path.abspath(""))
sys.path.append(os.path.abspath(""))

from MpuSimConv2D_gradient import *
from MpuSimConv2D import *

from MpuSimMatMul_gradient import *
from MpuSimFullyConnected import *

from imagenet_utils import ImageNetModel, get_imagenet_dataflow

session_conf = tf.ConfigProto(
      intra_op_parallelism_threads=1,
      inter_op_parallelism_threads=1)
sess = tf.Session(config=session_conf)


def visualize_conv1_weights(filters):
    ctx = get_current_tower_context()
    if not ctx.is_main_training_tower:
        return
    with tf.name_scope('visualize_conv1'):
        filters = tf.reshape(filters, [11, 11, 3, 8, 12])
        filters = tf.transpose(filters, [3, 0, 4, 1, 2])    # 8,11,12,11,3
        filters = tf.reshape(filters, [1, 88, 132, 3])
    tf.summary.image('visualize_conv1', filters, max_outputs=1, collections=['AAA'])


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
        with argscope([MpuSimConv2D, MaxPooling], data_format=self.data_format), \
                argscope([MpuSimConv2D, MpuSimFullyConnected],
                                            activation=tf.nn.relu,
                                            kernel_initializer=constant_init,
                                            activations_datatype_size_byte=self.activations_datatype_size_byte, 
                                            weights_datatype_size_byte=self.weights_datatype_size_byte,
                                            results_datatype_size_byte=self.results_datatype_size_byte,
                                            systolic_array_height=self.systolic_array_height,
                                            systolic_array_width=self.systolic_array_width,
                                            activation_fifo_depth=8,
                                            accumulator_array_height=self.accumulator_array_height,
                                            log_file_output_dir=("/home/kstehle/masters_thesis/"
                                                                    "tensorpack_models_mpusim/alexnet/"
                                                                    "mpu_log/width_height_sweep_constant_pe_count/"),
                                            model_name='alexnet_sys_arr_h_{}_sys_arr_w_{}_acc_arr_h_{}'.format(self.systolic_array_height,
                                                                                                                self.systolic_array_width, 
                                                                                                                self.accumulator_array_height)):
                
            # necessary padding to get 55x55 after conv1
            image = tf.pad(image, [[0, 0], [2, 2], [2, 2], [0, 0]])
            l = MpuSimConv2D('conv1', image, filters=96, kernel_size=11, strides=4, padding='VALID')
            # size: 55
            visualize_conv1_weights(l.variables.W)
#            l = tf.nn.lrn(l, 2, bias=1.0, alpha=2e-5, beta=0.75, name='norm1')
            l = MaxPooling('pool1', l, 3, strides=2, padding='VALID')
            # 27
            l = MpuSimConv2D('conv2', l, filters=256, kernel_size=5, split=2)
#            l = tf.nn.lrn(l, 2, bias=1.0, alpha=2e-5, beta=0.75, name='norm2')
            l = MaxPooling('pool2', l, 3, strides=2, padding='VALID')
            # 13
            l = MpuSimConv2D('conv3', l, filters=384, kernel_size=3)
            l = MpuSimConv2D('conv4', l, filters=384, kernel_size=3, split=2)
            l = MpuSimConv2D('conv5', l, filters=256, kernel_size=3, split=2)
            l = MaxPooling('pool3', l, 3, strides=2, padding='VALID')
            l = MpuSimFullyConnected('fc6', l, 4096,
                                        bias_initializer=tf.ones_initializer())
            l = MpuSimFullyConnected('fc7', l, 4096)
            return MpuSimFullyConnected('fc8', l, 1000)


def get_data(name, batch):
    isTrain = name == 'train'
    if isTrain:
        augmentors = [
            imgaug.ResizeShortestEdge(256, cv2.INTER_CUBIC),
            imgaug.RandomCrop(224),
            imgaug.Lighting(0.1,
                            eigval=np.asarray(
                                [0.2175, 0.0188, 0.0045][::-1]) * 255.0,
                            eigvec=np.array(
                                [[-0.5675, 0.7192, 0.4009],
                                 [-0.5808, -0.0045, -0.8140],
                                 [-0.5836, -0.6948, 0.4203]],
                                dtype='float32')[::-1, ::-1]),
            imgaug.Flip(horiz=True)]
    else:
        augmentors = [
            imgaug.ResizeShortestEdge(256, cv2.INTER_CUBIC),
            imgaug.CenterCrop((224, 224))]
    return get_imagenet_dataflow(args.data, name, batch, augmentors)


def get_config(activations_datatype_size_byte,
                weights_datatype_size_byte,
                results_datatype_size_byte,
                systolic_array_height,
                systolic_array_width,
                accumulator_array_height):
    
    nr_tower = max(get_num_gpu(), 1)
    BASE_LR = 0.01 * (1. / 128.)

    logger.info("Running on {} towers. Batch size per tower: {}".format(nr_tower, 1))
    
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
    
    logger.set_logger_dir(os.path.join('train_log', 'alexnet' + args.logdir_id))
    
    config = get_config(args.activations_datatype_size_byte,
                        args.weights_datatype_size_byte,
                        args.results_datatype_size_byte,
                        args.systolic_array_height,
                        args.systolic_array_width,
                        args.accumulator_array_height)
    
    launch_train_with_config(config, SimpleTrainer()) 