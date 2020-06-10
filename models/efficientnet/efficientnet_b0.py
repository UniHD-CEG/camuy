from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import copy
import functools
import os
import sys
import cv2
import tensorflow as tf
from tensorflow.contrib import slim

import conv_blocks

from tensorpack import *
from tensorpack.dataflow import imgaug
from tensorpack.tfutils import argscope
from tensorpack.utils.gpu import get_num_gpu

sys.path.append('../..')

from mpusim_conv2d.mpusim_conv2d_gradient import *
from mpusim_conv2d.mpusim_conv2d import *

from mpusim_separable_conv2d.mpusim_separable_conv2d_tensorpack import *

from mpusim_fc.mpusim_mat_mul_gradient import *
from mpusim_fc.mpusim_fully_connected import *

from models.imagenet_utils import ImageNetModel, get_imagenet_dataflow

# Disable parallel op execution to ensure that
# the MPU log outputs have the same order as
# the operations of the model

session_conf = tf.ConfigProto(
      intra_op_parallelism_threads=1,
      inter_op_parallelism_threads=1)
sess = tf.Session(config=session_conf)

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
                    accumulator_array_height=4096,
                    mpusim_logdir=''):
        super(Model, self).__init__(data_format, wd)

        self.activations_datatype_size_byte=activations_datatype_size_byte
        self.weights_datatype_size_byte=weights_datatype_size_byte
        self.results_datatype_size_byte=results_datatype_size_byte
        self.systolic_array_height=systolic_array_height
        self.systolic_array_width=systolic_array_width
        self.accumulator_array_height=accumulator_array_height

        self.mpusim_logdir=mpusim_logdir

    def get_logits(self, image):
        constant_init = tf.constant_initializer(1)
        with argscope([mpusim_conv2d,
                        mpusim_separable_convolution2d],
                        data_format=self.data_format,
                        activations_datatype_size_byte=self.activations_datatype_size_byte,
                        weights_datatype_size_byte=self.weights_datatype_size_byte,
                        results_datatype_size_byte=self.results_datatype_size_byte,
                        systolic_array_height=self.systolic_array_height,
                        systolic_array_width=self.systolic_array_width,
                        activation_fifo_depth=8,
                        accumulator_array_height=self.accumulator_array_height,
                        log_file_output_dir=self.mpusim_logdir,
                        model_name='mobilenet_v3_sys_arr_h_{}_sys_arr_w_{}'.format(self.systolic_array_height,
                                                                                    self.systolic_array_width)), \
                    argscope([mpusim_conv2d],
                                activation=tf.nn.relu,
                                kernel_initializer=constant_init):
                

def get_config(activations_datatype_size_byte,
                weights_datatype_size_byte,
                results_datatype_size_byte,
                systolic_array_height,
                systolic_array_width,
                accumulator_array_height,
                mpusim_logdir):
    
    nr_tower = max(get_num_gpu(), 1)
    BASE_LR = 0.01 * (1. / 128.)

    logger.info("Running on {} towers. Batch size per tower: {}".format(nr_tower, 1))
    
    data = QueueInput(FakeData(
            [[1, 224, 224, 3], [1]], 1, random=False, dtype='uint8'))
    callbacks = []

    return TrainConfig(
                model=Model(
                        activations_datatype_size_byte=activations_datatype_size_byte,
                        weights_datatype_size_byte=weights_datatype_size_byte,
                        results_datatype_size_byte=results_datatype_size_byte,
                        systolic_array_height=systolic_array_height,
                        systolic_array_width=systolic_array_width,
                        accumulator_array_height=accumulator_array_height,
                        mpusim_logdir=mpusim_logdir),
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
    parser.add_argument('--tensorpack-logdir-id', help='TensorPack training log directory id',
                            type=str, default='')
    parser.add_argument('--mpusim-logdir', help='MPU simulator log directory',
                            type=str, default='.')
    args = parser.parse_args()
    
    logger.set_logger_dir(os.path.join('train_log', 'mobilenet_v3' + args.tensorpack_logdir_id))
    
    config = get_config(args.activations_datatype_size_byte,
                        args.weights_datatype_size_byte,
                        args.results_datatype_size_byte,
                        args.systolic_array_height,
                        args.systolic_array_width,
                        args.accumulator_array_height,
                        args.mpusim_logdir)
    
    launch_train_with_config(config, SimpleTrainer())  
