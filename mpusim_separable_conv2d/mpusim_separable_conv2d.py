# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Keras convolution layers and image transformation layers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers.convolutional import SeparableConv
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec

from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.util.tf_export import tf_export

from . import mpusim_separable_conv2d_op_impl
    
class mpusim_separable_conv2d(SeparableConv):

    def __init__(self,
                    filters,
                    kernel_size,
                    strides=(1, 1),
                    padding='valid',
                    data_format=None,
                    dilation_rate=(1, 1),
                    depth_multiplier=1,
                    activation=None,
                    use_bias=True,
                    depthwise_initializer='glorot_uniform',
                    pointwise_initializer='glorot_uniform',
                    bias_initializer='zeros',
                    depthwise_regularizer=None,
                    pointwise_regularizer=None,
                    bias_regularizer=None,
                    activity_regularizer=None,
                    depthwise_constraint=None,
                    pointwise_constraint=None,
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
      
        super(mpusim_separable_conv2d, self).__init__(
                                                    rank=2,
                                                    filters=filters,
                                                    kernel_size=kernel_size,
                                                    strides=strides,
                                                    padding=padding,
                                                    data_format=data_format,
                                                    dilation_rate=dilation_rate,
                                                    depth_multiplier=depth_multiplier,
                                                    activation=activations.get(activation),
                                                    use_bias=use_bias,
                                                    depthwise_initializer=initializers.get(depthwise_initializer),
                                                    pointwise_initializer=initializers.get(pointwise_initializer),
                                                    bias_initializer=initializers.get(bias_initializer),
                                                    depthwise_regularizer=regularizers.get(depthwise_regularizer),
                                                    pointwise_regularizer=regularizers.get(pointwise_regularizer),
                                                    bias_regularizer=regularizers.get(bias_regularizer),
                                                    activity_regularizer=regularizers.get(activity_regularizer),
                                                    depthwise_constraint=constraints.get(depthwise_constraint),
                                                    pointwise_constraint=constraints.get(pointwise_constraint),
                                                    bias_constraint=constraints.get(bias_constraint),
                                                    **kwargs)
        
        self.activations_datatype_size_byte=activations_datatype_size_byte
        self.weights_datatype_size_byte=weights_datatype_size_byte
        self.results_datatype_size_byte=results_datatype_size_byte
        self.systolic_array_height=systolic_array_height
        self.systolic_array_width=systolic_array_width
        self.activation_fifo_depth=activation_fifo_depth
        self.accumulator_array_height=accumulator_array_height
        self.log_file_output_dir=log_file_output_dir
        self.model_name=model_name

    def call(self, inputs):
        
        # Apply the actual ops
        
        if self.data_format is not 'channels_last':
            raise ValueError("mpusim_separable_conv2d "
                                "requires NHWC data format")
            
        strides = (1,) + self.strides + (1,)

        outputs = mpusim_separable_conv2d_op_impl(inputs,
                                                    self.depthwise_kernel,
                                                    self.pointwise_kernel,
                                                    strides,
                                                    self.padding.upper(),
                                                    self.dilation_rate,
                                                    None,
                                                    conv_utils.convert_data_format(self.data_format, ndim=4),
                                                    self.activations_datatype_size_byte,
                                                    self.weights_datatype_size_byte,
                                                    self.results_datatype_size_byte,
                                                    self.systolic_array_height,
                                                    self.systolic_array_width,
                                                    self.activation_fifo_depth,
                                                    self.accumulator_array_height,
                                                    self.log_file_output_dir,
                                                    self.model_name)

        if self.use_bias:
            outputs = nn.bias_add(outputs,
                                    self.bias,
                                    data_format= \
                                        conv_utils.convert_data_format(self.data_format, ndim=4))

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

