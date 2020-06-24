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

import numpy as np
from tensorpack.compat import tfv1 as tf  # this should be avoided first in model code

from tensorpack.tfutils.common import get_tf_version_tuple
from tensorpack.models.common import VariableHolder, layer_register
from tensorpack.models.tflayer import convert_to_tflayer_args, rename_get_variable

from .mpusim_fc import *

__all__ = ['mpusim_fully_connected']
def batch_flatten(x):
    """
    Flatten the tensor except the first dimension.
    """
    shape = x.get_shape().as_list()[1:]
    if None not in shape:
        return tf.reshape(x, [-1, int(np.prod(shape))])
    return tf.reshape(x, tf.stack([tf.shape(x)[0], -1]))


@layer_register(log_shape=True)
@convert_to_tflayer_args(
    args_names=['units'],
    name_mapping={'out_dim': 'units'})
def mpusim_fully_connected(inputs,
                            units,
                            activation=None,
                            use_bias=True,
                            kernel_initializer=None,
                            bias_initializer=tf.zeros_initializer(),
                            kernel_regularizer=None,
                            bias_regularizer=None,
                            activity_regularizer=None,
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
    A wrapper around `mpusim_fc`.
    One difference to maintain backward-compatibility:
    Default weight initializer is variance_scaling_initializer(2.0).
    Variable Names:
    * ``W``: weights of shape [in_dim, out_dim]
    * ``b``: bias
    """
    if kernel_initializer is None:
        if get_tf_version_tuple() <= (1, 12):
            kernel_initializer = tf.contrib.layers.variance_scaling_initializer(2.0)  # deprecated
        else:
            kernel_initializer = tf.keras.initializers.VarianceScaling(2.0, distribution='untruncated_normal')

    inputs = batch_flatten(inputs)
    with rename_get_variable({'kernel': 'W', 'bias': 'b'}):
        layer = mpusim_fc(units=units,
                            activation=activation,
                            use_bias=use_bias,
                            kernel_initializer=kernel_initializer,
                            bias_initializer=bias_initializer,
                            kernel_regularizer=kernel_regularizer,
                            bias_regularizer=bias_regularizer,
                            activity_regularizer=activity_regularizer,
                            activations_datatype_size_byte=activations_datatype_size_byte,
                            weights_datatype_size_byte=weights_datatype_size_byte,
                            results_datatype_size_byte=results_datatype_size_byte,
                            systolic_array_height=systolic_array_height,
                            systolic_array_width=systolic_array_width,
                            activation_fifo_depth=activation_fifo_depth,
                            accumulator_array_height=accumulator_array_height,
                            log_file_output_dir=log_file_output_dir,
                            model_name=model_name,
                            _reuse=tf.get_variable_scope().reuse)
        ret = layer.apply(inputs, scope=tf.get_variable_scope())
        ret = tf.identity(ret, name='output')

    ret.variables = VariableHolder(W=layer.kernel)
    
    if use_bias:
        ret.variables.b = layer.bias
    return ret

