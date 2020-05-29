
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

def hard_swish(x):
    with tf.compat.v1.name_scope('hard_swish'):
        return x*tf.nn.relu6(x + np.float32(3))*np.float32(1./6.)
    
def reduce_to_1x1(input_tensor, default_size=7, **kwargs):
    
    h, w = input_tensor.shape.as_list()[1:3]
    if h is not None and w == h:
        k = [h, h]
    else:
        k = [default_size, default_size]
    return slim.avg_pool2d(input_tensor, kernel_size=k, **kwargs)
    
def mbv3_op(input_tensor, ef, n, k, s=1, act=tf.nn.relu, se=None, **kwargs):
    
    return conv_blocks.expanded_conv(input_tensor,
                                        expansion_size=conv_blocks.expand_input_by_factor(ef),
                                        kernel_size=(k, k),
                                        stride=s,
                                        num_outputs=n,
                                        inner_activation_fn=act,
                                        expansion_transform=se,
                                        **kwargs)

# Squeeze Excite with all parameters filled-in, we use hard-sigmoid
# for gating function and relu for inner activation function.

squeeze_excite = functools.partial(
    conv_blocks.squeeze_excite, squeeze_factor=4,
    inner_activation_fn=tf.nn.relu,
    gating_fn=lambda x: tf.nn.relu6(x+3)*0.16667)

# Wrap squeeze excite op as expansion_transform that takes
# both expansion and input tensor.

_se4 = lambda expansion_tensor, input_tensor: squeeze_excite(expansion_tensor)

mbv3_op_se = functools.partial(mbv3_op, se=_se4)

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
        with argscope([mpusim_conv2d],
                        data_format=self.data_format,
                        activation=tf.nn.relu,
                        kernel_initializer=constant_init):
                
            
            l = mpusim_conv2d('Conv', image, 16,
                                3, strides=(2, 2), activation=hard_swish)
            
            l = mbv3_op_se(l, ef=1, n=16, k=3, s=2)
            l = mbv3_op(l, ef=72./16, n=24, k=3, s=2)
            l = mbv3_op(l, ef=(88./24), n=24, k=3, s=1)
            l = mbv3_op_se(l, ef=4, n=40, k=5, s=2, act=hard_swish)
            l = mbv3_op_se(l, ef=6, n=40, k=5, s=1, act=hard_swish)
            l = mbv3_op_se(l, ef=6, n=40, k=5, s=1, act=hard_swish)
            l = mbv3_op_se(l, ef=3, n=48, k=5, s=1, act=hard_swish)
            l = mbv3_op_se(l, ef=3, n=48, k=5, s=1, act=hard_swish)
            l = mbv3_op_se(l, ef=6, n=96, k=5, s=2, act=hard_swish)
            l = mbv3_op_se(l, ef=6, n=96, k=5, s=1, act=hard_swish)
            l = mbv3_op_se(l, ef=6, n=96, k=5, s=1, act=hard_swish)
            
            l = mpusim_conv2d('Conv_1', l, 576, 1,
                                activation=hard_swish)
            
            l = reduce_to_1x1(l, default_size=7, stride=1, padding='VALID')
            
            l = mpusim_conv2d('Conv_2', l, 1024, 1,
                                activation=hard_swish)
            
            l = mpusim_conv2d('Conv2d_1c_1x1', l, 1000, 1, activation=None,
                                bias_initializer=tf.compat.v1.zeros_initializer())

            return tf.squeeze(l, [1, 2])


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
