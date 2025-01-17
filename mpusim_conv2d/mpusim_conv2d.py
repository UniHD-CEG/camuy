# Copyright Yuxin Wu
# Modifications copyright (c) 2020 Computing Systems Group
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops

from tensorpack.compat import tfv1 as tf

from tensorpack.tfutils.common import get_tf_version_tuple
from tensorpack.utils.argtools import get_data_format, shape2d, shape4d, log_once
from tensorpack.models.common import VariableHolder, layer_register
from tensorpack.models.tflayer import convert_to_tflayer_args, rename_get_variable

mpu_sim_conv2d_lib = tf.load_op_library('../../bin/build_mpusim_conv2d_release/mpusim-conv2d.so')

__all__ = ['mpusim_conv2d']

@layer_register(log_shape=True)
@convert_to_tflayer_args(
    args_names=['filters', 'kernel_size'],
    name_mapping={
        'out_channel': 'filters',
        'kernel_shape': 'kernel_size',
        'stride': 'strides',
    })
def mpusim_conv2d(
        inputs,
        filters,
        kernel_size,
        strides=(1, 1),
        padding='same',
        data_format='channels_last',
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        split=1,
        activations_datatype_size_byte=1,
        weights_datatype_size_byte=1,
        results_datatype_size_byte=4,
        systolic_array_height=256,
        systolic_array_width=256,
        activation_fifo_depth=8,
        accumulator_array_height=4096,
        log_file_output_dir='.',
        model_name='unnamed'):
    """
    Similar to `tf.layers.Conv2D`, but with some differences:

    1. Default kernel initializer is variance_scaling_initializer(2.0).
    2. Default padding is 'same'.
    3. Support 'split' argument to do group convolution.

    Variable Names:

    * ``W``: weights
    * ``b``: bias
    """
    if kernel_initializer is None:
        if get_tf_version_tuple() <= (1, 12):
            kernel_initializer = tf.contrib.layers.variance_scaling_initializer(2.0)
        else:
            kernel_initializer = tf.keras.initializers.VarianceScaling(2.0, distribution='untruncated_normal')
    dilation_rate = shape2d(dilation_rate)

    # group conv implementation
    data_format = get_data_format(data_format, keras_mode=False)
    in_shape = inputs.get_shape().as_list()
    channel_axis = 3 if data_format == 'NHWC' else 1
    in_channel = in_shape[channel_axis]
    assert in_channel is not None, "[mpusim_conv2d] Input cannot have unknown channel!"
    assert in_channel % split == 0

    assert kernel_regularizer is None and bias_regularizer is None and activity_regularizer is None, \
        "Not supported by group conv or dilated conv!"

    out_channel = filters
    assert out_channel % split == 0
    assert dilation_rate == [1, 1] or get_tf_version_tuple() >= (1, 5), 'TF>=1.5 required for dilated conv.'

    kernel_shape = shape2d(kernel_size)
    filter_shape = kernel_shape + [in_channel / split, out_channel]
    stride = shape4d(strides, data_format=data_format)

    kwargs = dict(data_format=data_format)
    if get_tf_version_tuple() >= (1, 5):
        kwargs['dilations'] = shape4d(dilation_rate, data_format=data_format)

    W = tf.get_variable(
            'W', filter_shape, initializer=kernel_initializer)

    if use_bias:
        b = tf.get_variable('b', [out_channel], initializer=bias_initializer)

    if split == 1:
        conv = mpu_sim_conv2d_lib.mpu_sim_conv2d(inputs,
                                                    W,
                                                    activations_datatype_size_byte,
                                                    weights_datatype_size_byte,
                                                    results_datatype_size_byte,
                                                    systolic_array_height,
                                                    systolic_array_width,
                                                    activation_fifo_depth,
                                                    accumulator_array_height,
                                                    log_file_output_dir,
                                                    model_name,
                                                    stride,
                                                    padding.upper(),
                                                    **kwargs)
    else:
        
        inputs = tf.split(inputs, split, channel_axis)
        kernels = tf.split(W, split, 3)
        outputs = [mpu_sim_conv2d_lib.mpu_sim_conv2d(input_block,
                                                        kernel_block,
                                                        activations_datatype_size_byte,
                                                        weights_datatype_size_byte,
                                                        results_datatype_size_byte,
                                                        systolic_array_height,
                                                        systolic_array_width,
                                                        activation_fifo_depth,
                                                        accumulator_array_height,
                                                        log_file_output_dir,
                                                        model_name,
                                                        stride,
                                                        padding.upper(),
                                                        **kwargs)
                    for input_block, kernel_block in zip(inputs, kernels)]
        conv = tf.concat(outputs, channel_axis)

    ret = tf.nn.bias_add(conv, b, data_format=data_format) if use_bias else conv
    if activation is not None:
        ret = activation(ret)
    ret = tf.identity(ret, name='output')

    ret.variables = VariableHolder(W=W)
    if use_bias:
        ret.variables.b=b
    return ret

