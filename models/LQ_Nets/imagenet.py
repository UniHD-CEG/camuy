#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: imagenet.py

import argparse
import os
import sys
import tensorflow as tf

from tensorpack import *
from tensorpack.callbacks import *
from tensorpack.dataflow import FakeData
from tensorpack.models import *
from tensorpack.tfutils import argscope, get_model_loader
from tensorpack.tfutils.summary import *
from tensorpack.train import AutoResumeTrainConfig, SyncMultiGPUTrainerReplicated, launch_train_with_config
from tensorpack.utils.gpu import get_nr_gpu

session_conf = tf.ConfigProto(
      intra_op_parallelism_threads=1,
      inter_op_parallelism_threads=1)
sess = tf.Session(config=session_conf)

sys.path.append('../..')

from mpusim_conv2d.mpusim_conv2d_gradient import *
from mpusim_conv2d.mpusim_conv2d import *

from mpusim_fc.mpusim_mat_mul_gradient import *
from mpusim_fc.mpusim_fully_connected import *

import imagenet_utils
from densenet_model import densenet_backbone
from googlenet_model import googlenet_backbone
from imagenet_utils import (
    fbresnet_augmentor, normal_augmentor, get_imagenet_dataflow, ImageNetModel,
    eval_on_ILSVRC12)
from resnet_model import (
  preresnet_group, preresnet_group_typeA, preresnet_basicblock, preresnet_bottleneck,
  resnet_group, resnet_basicblock, resnet_bottleneck, resnet_backbone)

TOTAL_BATCH_SIZE = 1

class Model(ImageNetModel):

    def inputs(self):
        return [tf.TensorSpec([None, 224, 224, 3], tf.float32, 'input'),
                tf.TensorSpec([None], tf.int32, 'label')]

    def __init__(self,
                    mode,
                    resnet_depth,
                    activations_datatype_size_byte=1,
                    weights_datatype_size_byte=1,
                    results_datatype_size_byte=4,
                    systolic_array_height=256,
                    systolic_array_width=256,
                    accumulator_array_height=4096,
                    mpusim_logdir='.'):
        super(Model, self).__init__('NHWC', 5e-5, 0.1, True, double_iter=False)

        self.mode=mode
        self.resnet_depth=resnet_depth
        
        self.activations_datatype_size_byte=activations_datatype_size_byte
        self.weights_datatype_size_byte=weights_datatype_size_byte
        self.results_datatype_size_byte=results_datatype_size_byte
        self.systolic_array_height=systolic_array_height
        self.systolic_array_width=systolic_array_width
        self.accumulator_array_height=accumulator_array_height
        
        self.mpusim_logdir=mpusim_logdir
        
        if mode == 'vgg' or mode == 'alexnet' or mode == 'googlenet' or mode == 'densenet':
            return
        basicblock = preresnet_basicblock if mode == 'preact' else resnet_basicblock
        bottleneck = {
            'resnet': resnet_bottleneck,
            'preact': preresnet_bottleneck,
            'preact_typeA': preresnet_bottleneck}[mode]
        self.num_blocks, self.block_func = {
            18: ([2, 2, 2, 2], basicblock),
            34: ([3, 4, 6, 3], basicblock),
            50: ([3, 4, 6, 3], bottleneck),
            101: ([3, 4, 23, 3], bottleneck),
            152: ([3, 8, 36, 3], bottleneck)
        }[resnet_depth]

    def build_graph(self, image, label):
        with argscope([mpusim_conv2d, MaxPooling, AvgPooling, GlobalAvgPooling, BatchNorm], data_format=self.data_format):
            if self.mode == 'googlenet':
                l =  googlenet_backbone(image,
                                        self.activations_datatype_size_byte,
                                        self.weights_datatype_size_byte,
                                        self.results_datatype_size_byte,
                                        self.systolic_array_height,
                                        self.systolic_array_width,
                                        self.accumulator_array_height,
                                        self.mpusim_logdir)
            elif self.mode == 'densenet':
                l =  densenet_backbone(image,
                                        self.activations_datatype_size_byte,
                                        self.weights_datatype_size_byte,
                                        self.results_datatype_size_byte,
                                        self.systolic_array_height,
                                        self.systolic_array_width,
                                        self.accumulator_array_height,
                                        self.mpusim_logdir)
            else:
                group_func = resnet_group
                l = resnet_backbone(image,
                                        self.resnet_depth,
                                        self.num_blocks,
                                        group_func,
                                        self.block_func,
                                        self.activations_datatype_size_byte,
                                        self.weights_datatype_size_byte,
                                        self.results_datatype_size_byte,
                                        self.systolic_array_height,
                                        self.systolic_array_width,
                                        self.accumulator_array_height,
                                        self.mpusim_logdir)
            
            tf.nn.softmax(l, name='output')
            loss3 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=l, labels=label)
            loss3 = tf.reduce_mean(loss3, name='loss3')

            cost = tf.add_n([loss3, 0.3, 0.3], name='weighted_cost')

            def prediction_incorrect(logits, label, topk, name):
                return tf.cast(tf.logical_not(tf.nn.in_top_k(logits, label, topk)), tf.float32, name=name)

            wrong = prediction_incorrect(l, label, 1, name='wrong-top1')
            wrong = prediction_incorrect(l, label, 5, name='wrong-top5')
            
            wd_cost = tf.multiply(0.5, regularize_cost('.*/W', tf.nn.l2_loss), name='l2_regularize_loss')

            total_cost = tf.add_n([cost, wd_cost], name='cost')
            return total_cost


    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.045, trainable=False)
        return tf.train.MomentumOptimizer(lr, 0.9)


def get_config(model):
    batch = TOTAL_BATCH_SIZE

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
    parser.add_argument('--mode', choices=['resnet', 'googlenet', 'densenet'],
                        help='Type of model used for ',
                        default='resnet')
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
    
    imagenet_utils.DEFAULT_IMAGE_SHAPE = 224

    model = Model(args.mode,
                    args.resnet_depth,
                    args.activations_datatype_size_byte,
                    args.weights_datatype_size_byte,
                    args.results_datatype_size_byte,
                    args.systolic_array_height,
                    args.systolic_array_width,
                    args.accumulator_array_height,
                    args.mpusim_logdir)

    logger.set_logger_dir(os.path.join('train_log', 'imagenet' + args.tensorpack_logdir_id))

    config = get_config(model)
    launch_train_with_config(config, SimpleTrainer())
