#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: imagenet-resnet.py

import argparse
import os
import sys
import tensorflow as tf

from tensorpack import QueueInput, TFDatasetInput, logger
from tensorpack.callbacks import *
from tensorpack.dataflow import FakeData
from tensorpack.models import *
from tensorpack.tfutils import argscope
from tensorpack.train import SimpleTrainer, TrainConfig, launch_train_with_config
from tensorpack.utils.gpu import get_num_gpu

session_conf = tf.ConfigProto(
      intra_op_parallelism_threads=1,
      inter_op_parallelism_threads=1)
sess = tf.Session(config=session_conf)

sys.path.append('../..')

from mpusim_conv2d.mpusim_conv2d_gradient import *
from mpusim_conv2d.mpusim_conv2d import *

from mpusim_fc.mpusim_mat_mul_gradient import *
from mpusim_fc.mpusim_fully_connected import *

from models.imagenet_utils import ImageNetModel, eval_classification, get_imagenet_dataflow, get_imagenet_tfdata
import resnext_model
from resnext_model import preact_group, resnet_backbone, resnet_group


class Model(ImageNetModel):
    def __init__(self,
                    resnet_depth,
                    activations_datatype_size_byte=1,
                    weights_datatype_size_byte=1,
                    results_datatype_size_byte=4,
                    systolic_array_height=256,
                    systolic_array_width=256,
                    accumulator_array_height=4096,
                    mpusim_logdir='.'):

        self.resnet_depth=resnet_depth
        
        self.activations_datatype_size_byte=activations_datatype_size_byte
        self.weights_datatype_size_byte=weights_datatype_size_byte
        self.results_datatype_size_byte=results_datatype_size_byte
        self.systolic_array_height=systolic_array_height
        self.systolic_array_width=systolic_array_width
        self.accumulator_array_height=accumulator_array_height
        
        self.mpusim_logdir=mpusim_logdir        
        
        basicblock = getattr(resnext_model, 'resnext32x4d_basicblock', None)
        bottleneck = getattr(resnext_model, 'resnext32x4d_bottleneck', None)
        self.num_blocks, self.block_func = {
            18: ([2, 2, 2, 2], basicblock),
            34: ([3, 4, 6, 3], basicblock),
            50: ([3, 4, 6, 3], bottleneck),
            101: ([3, 4, 23, 3], bottleneck),
            152: ([3, 8, 36, 3], bottleneck)
        }[self.resnet_depth]
        assert self.block_func is not None, \
            "(mode={}, resnet_depth={}) not implemented!".format('resnext32x4d', self.resnet_depth)

    def get_logits(self, image):
        constant_init = tf.constant_initializer(1)
        
        with argscope([mpusim_conv2d, MaxPooling, GlobalAvgPooling, BatchNorm], data_format='NHWC'), \
                argscope([mpusim_conv2d, mpusim_fully_connected],
                                            kernel_initializer=constant_init,
                                            activations_datatype_size_byte=self.activations_datatype_size_byte, 
                                            weights_datatype_size_byte=self.weights_datatype_size_byte,
                                            results_datatype_size_byte=self.results_datatype_size_byte,
                                            systolic_array_height=self.systolic_array_height,
                                            systolic_array_width=self.systolic_array_width,
                                            activation_fifo_depth=8,
                                            accumulator_array_height=self.accumulator_array_height,
                                            log_file_output_dir=self.mpusim_logdir,
                                            model_name='resnext_{}_sys_arr_w_{}_acc_arr_h_{}'.format(self.resnet_depth,
                                                                                                        self.systolic_array_height,
                                                                                                                self.systolic_array_width)):
            return resnet_backbone(image,
                                    self.num_blocks,
                                    resnet_group,
                                    self.block_func)

def get_config(model):
    batch = 1

    logger.info("For benchmark, batch size is fixed to 1 per tower.")
    data = QueueInput(FakeData(
            [[1, 224, 224, 3], [1]], 1, random=False, dtype='uint8'))

    return TrainConfig(
                model=model,
                data=data,
                callbacks=[],
                steps_per_epoch=1,
                max_epoch=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--resnet-depth', help='resnet depth',
                        type=int, default=18, choices=[18, 34, 50, 101, 152])
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

    model = Model(args.resnet_depth,
                    args.activations_datatype_size_byte,
                    args.weights_datatype_size_byte,
                    args.results_datatype_size_byte,
                    args.systolic_array_height,
                    args.systolic_array_width,
                    args.accumulator_array_height,
                    args.mpusim_logdir)

    logger.set_logger_dir(os.path.join('train_log', 'resnext_{}{}'.format(args.resnet_depth,
                                                                            args.tensorpack_logdir_id)))

    config = get_config(model)
    launch_train_with_config(config, SimpleTrainer())

