# Copyright Yuxin Wu
# Modifications copyright 2020 Kevin Stehle
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import sys
import os
import tensorflow as tf

from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils.gpu import get_num_gpu

sys.path.append('../..')

from mpusim_conv2d.mpusim_conv2d_gradient import *
from mpusim_conv2d.mpusim_conv2d import *

from mpusim_fc.mpusim_mat_mul_gradient import *
from mpusim_fc.mpusim_fully_connected import *

from models.imagenet_utils import fbresnet_augmentor, get_imagenet_dataflow, ImageNetModel

session_conf = tf.ConfigProto(
      intra_op_parallelism_threads=1,
      inter_op_parallelism_threads=1)
sess = tf.Session(config=session_conf)

INPUT_SHAPE = 224

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

    def inputs(self):
        return [tf.TensorSpec([None, INPUT_SHAPE, INPUT_SHAPE, 3], tf.float32, 'input'),
                tf.TensorSpec([None], tf.int32, 'label')]

    def build_graph(self, image, label):
        image = image / 128.0

        def inception(name, x, nr1x1, nr3x3r, nr3x3, nr233r, nr233, nrpool, pooltype):
            stride = 2 if nr1x1 == 0 else 1
            with tf.variable_scope(name):
                outs = []
                if nr1x1 != 0:
                    outs.append(mpusim_conv2d('conv1x1', x, nr1x1, 1))
                x2 = mpusim_conv2d('conv3x3r', x, nr3x3r, 1)
                outs.append(mpusim_conv2d('conv3x3', x2, nr3x3, 3, strides=stride))

                x3 = mpusim_conv2d('conv233r', x, nr233r, 1)
                x3 = mpusim_conv2d('conv233a', x3, nr233, 3)
                outs.append(mpusim_conv2d('conv233b', x3, nr233, 3, strides=stride))

                if pooltype == 'max':
                    x4 = MaxPooling('mpool', x, 3, stride, padding='SAME')
                else:
                    assert pooltype == 'avg'
                    x4 = AvgPooling('apool', x, 3, stride, padding='SAME')
                if nrpool != 0:  # pool + passthrough if nrpool == 0
                    x4 = mpusim_conv2d('poolproj', x4, nrpool, 1)
                outs.append(x4)
                return tf.concat(outs, 3, name='concat')
            
        constant_init = tf.constant_initializer(1)

        with argscope(mpusim_conv2d,
                        activation=BNReLU,
                        use_bias=False,
                        data_format=self.data_format), \
                argscope([mpusim_conv2d, mpusim_fully_connected],
                                            activation=tf.nn.relu,
                                            kernel_initializer=constant_init,
                                            activations_datatype_size_byte=self.activations_datatype_size_byte, 
                                            weights_datatype_size_byte=self.weights_datatype_size_byte,
                                            results_datatype_size_byte=self.results_datatype_size_byte,
                                            systolic_array_height=self.systolic_array_height,
                                            systolic_array_width=self.systolic_array_width,
                                            activation_fifo_depth=8,
                                            accumulator_array_height=self.accumulator_array_height,
                                            log_file_output_dir=self.mpusim_logdir,
                                            model_name='inception_bn_sys_arr_h_{}_sys_arr_w_{}_acc_arr_h_{}'.format(self.systolic_array_height,
                                                                                                                self.systolic_array_width, 
                                                                                                                self.accumulator_array_height)):
            l = mpusim_conv2d('conv0', image, 64, 7, strides=2)
            l = MaxPooling('pool0', l, 3, 2, padding='SAME')
            l = mpusim_conv2d('conv1', l, 64, 1)
            l = mpusim_conv2d('conv2', l, 192, 3)
            l = MaxPooling('pool2', l, 3, 2, padding='SAME')
            # 28
            l = inception('incep3a', l, 64, 64, 64, 64, 96, 32, 'avg')
            l = inception('incep3b', l, 64, 64, 96, 64, 96, 64, 'avg')
            l = inception('incep3c', l, 0, 128, 160, 64, 96, 0, 'max')

            # 14
            l = inception('incep4a', l, 224, 64, 96, 96, 128, 128, 'avg')
            l = inception('incep4b', l, 192, 96, 128, 96, 128, 128, 'avg')
            l = inception('incep4c', l, 160, 128, 160, 128, 160, 128, 'avg')
            l = inception('incep4d', l, 96, 128, 192, 160, 192, 128, 'avg')
            l = inception('incep4e', l, 0, 128, 192, 192, 256, 0, 'max')

            # 7
            l = inception('incep5a', l, 352, 192, 320, 160, 224, 128, 'avg')
            l = inception('incep5b', l, 352, 192, 320, 192, 224, 128, 'max')
            l = GlobalAvgPooling('gap', l)

            logits = mpusim_fully_connected('linear', l, 1000, activation=tf.identity)
        tf.nn.softmax(logits, name='output')
        
        loss3 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        loss3 = tf.reduce_mean(loss3, name='loss3')

        cost = tf.add_n([loss3, 0.3, 0.3], name='weighted_cost')

        def prediction_incorrect(logits, label, topk, name):
            return tf.cast(tf.logical_not(tf.nn.in_top_k(logits, label, topk)), tf.float32, name=name)

        wrong = prediction_incorrect(logits, label, 1, name='wrong-top1')
        wrong = prediction_incorrect(logits, label, 5, name='wrong-top5')

        wd_cost = tf.multiply(0.5, regularize_cost('.*/W', tf.nn.l2_loss), name='l2_regularize_loss')

        total_cost = tf.add_n([cost, wd_cost], name='cost')
        add_moving_summary(wd_cost, total_cost)
        return total_cost

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.045, trainable=False)
        return tf.train.MomentumOptimizer(lr, 0.9)


def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    augs = fbresnet_augmentor(isTrain)

    meta = dataset.ILSVRCMeta()
    pp_mean = meta.get_per_pixel_mean()
    augs.append(imgaug.MapImage(lambda x: x - pp_mean[16:-16, 16:-16]))

    ds = get_imagenet_dataflow(args.data, train_or_test, 1, augs)
    return ds


def get_config(activations_datatype_size_byte,
                weights_datatype_size_byte,
                results_datatype_size_byte,
                systolic_array_height,
                systolic_array_width,
                accumulator_array_height,
                mpusim_logdir):

    data = QueueInput(FakeData(
            [[1, 224, 224, 3], [1]], 1, random=False, dtype='uint8'))

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
                callbacks=[],
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


    logger.set_logger_dir(os.path.join('train_log', 'inception_bn' + args.tensorpack_logdir_id))
    
    config = get_config(args.activations_datatype_size_byte,
                        args.weights_datatype_size_byte,
                        args.results_datatype_size_byte,
                        args.systolic_array_height,
                        args.systolic_array_width,
                        args.accumulator_array_height,
                        args.mpusim_logdir)

    launch_train_with_config(config, SimpleTrainer()) 
