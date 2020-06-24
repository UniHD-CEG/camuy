# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

import copy
import sys
import types as python_types
import warnings

import numpy as np

from tensorpack.compat import tfv1 as tf  # this should be avoided first in model code

from tensorflow.python.eager import context
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import layers as keras_layers
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.layers import base
from tensorflow.python.ops import init_ops

mpu_sim_mat_mul_lib = tf.load_op_library('../../bin/build_mpusim_mat_mul_release/mpusim-mat-mul.so')

class mpusim_fc_base(Layer):
    
    def __init__(self,
                    units,
                    activation=None,
                    use_bias=True,
                    kernel_initializer='glorot_uniform',
                    bias_initializer='zeros',
                    kernel_regularizer=None,
                    bias_regularizer=None,
                    activity_regularizer=None,
                    kernel_constraint=None,
                    bias_constraint=None,
                    activations_datatype_size_byte=1,
                    weights_datatype_size_byte=1,
                    results_datatype_size_byte=4,
                    systolic_array_height=256,
                    systolic_array_width=256,
                    activation_fifo_depth=8,
                    accumulator_array_height=4096,
                    log_file_output_dir='.',
                    model_name='unnamed',
                    **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(mpusim_fc_base, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer), **kwargs)
        self.units = int(units)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.supports_masking = True
        self.input_spec = InputSpec(min_ndim=2)
        
        self.activations_datatype_size_byte=activations_datatype_size_byte
        self.weights_datatype_size_byte=weights_datatype_size_byte
        self.results_datatype_size_byte=results_datatype_size_byte
        self.systolic_array_height=systolic_array_height
        self.systolic_array_width=systolic_array_width
        self.activation_fifo_depth=activation_fifo_depth
        self.accumulator_array_height=accumulator_array_height
        self.log_file_output_dir=log_file_output_dir
        self.model_name=model_name

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError('The last dimension of the inputs to `mpusim_fc_base` '
                            'should be defined. Found `None`.')
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        self.input_spec = InputSpec(min_ndim=2,
                                    axes={-1: last_dim})
        self.kernel = self.add_weight(
            'kernel',
            shape=[last_dim, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(
                'bias',
                shape=[self.units,],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        inputs = ops.convert_to_tensor(inputs)
        rank = common_shapes.rank(inputs)
        if rank > 2:
            raise ValueError('mpusim_fc_base only supports tensors of rank <= 2')
        else:
            
            outputs = mpu_sim_mat_mul_lib.mpu_sim_mat_mul(inputs,
                                                            self.kernel,
                                                            activations_datatype_size_byte=self.activations_datatype_size_byte,
                                                            weights_datatype_size_byte=self.weights_datatype_size_byte,
                                                            results_datatype_size_byte=self.results_datatype_size_byte,
                                                            systolic_array_height=self.systolic_array_height,
                                                            systolic_array_width=self.systolic_array_width,
                                                            activation_fifo_depth=self.activation_fifo_depth,
                                                            accumulator_array_height=self.accumulator_array_height,
                                                            log_file_output_dir=self.log_file_output_dir,
                                                            model_name=self.model_name)

        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % input_shape)
        return input_shape[:-1].concatenate(self.units)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(mpusim_fc_base, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class mpusim_fc(mpusim_fc_base, base.Layer):
    
    def __init__(self, 
                    units,
                    activation=None,
                    use_bias=True,
                    kernel_initializer=None,
                    bias_initializer=init_ops.zeros_initializer(),
                    kernel_regularizer=None,
                    bias_regularizer=None,
                    activity_regularizer=None,
                    kernel_constraint=None,
                    bias_constraint=None,
                    trainable=True,
                    name=None,
                    **kwargs):
        super(mpusim_fc, self).__init__(units=units,
                                        activation=activation,
                                        use_bias=use_bias,
                                        kernel_initializer=kernel_initializer,
                                        bias_initializer=bias_initializer,
                                        kernel_regularizer=kernel_regularizer,
                                        bias_regularizer=bias_regularizer,
                                        activity_regularizer=activity_regularizer,
                                        kernel_constraint=kernel_constraint,
                                        bias_constraint=bias_constraint,
                                        trainable=trainable,
                                        name=name,
                                        **kwargs)
