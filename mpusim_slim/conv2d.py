
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers

import numpy as np

from tensorflow.python.compat import compat
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util

from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec

from tensorflow.python.keras.layers.pooling import AveragePooling1D
from tensorflow.python.keras.layers.pooling import AveragePooling2D
from tensorflow.python.keras.layers.pooling import AveragePooling3D
from tensorflow.python.keras.layers.pooling import MaxPooling1D
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.python.keras.layers.pooling import MaxPooling3D

from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import tf_utils

from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops

from tensorflow.python.ops.gen_nn_ops import *

from tensorflow.python.util import deprecation
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup
from tensorflow.python.util.tf_export import tf_export

local_response_normalization = gen_nn_ops.lrn

mpu_sim_conv2d_lib = tf.load_op_library('../../bin/build_mpusim_conv2d_release/mpusim-conv2d.so')

__all__ = ["Conv2D"]

class _NonAtrousConvolution(object):
    def __init__(self,
                input_shape,
                filter_shape,
                padding,
                data_format=None,
                strides=None,
                name=None):
      
        filter_shape = filter_shape.with_rank(input_shape.ndims)
        
        self.padding = padding
        self.name = name
        
        input_shape = input_shape.with_rank(filter_shape.ndims)
        
        if input_shape.ndims is None:
            raise ValueError("Rank of convolution must be known")
        
        if input_shape.ndims < 3 or input_shape.ndims > 5:
            raise ValueError("`input` and `filter` must have rank at least 3 and at most 5")
        
        conv_dims = input_shape.ndims - 2
        
        if strides is None:
            strides = [1] * conv_dims
            
        elif len(strides) != conv_dims:
            raise ValueError("len(strides)=%d, but should be %d" % (len(strides), conv_dims))
        
        if conv_dims == 1:
            raise ValueError('mpusim_slim._NonAtrousConvolution currently only supports 2D convolution')

            #if data_format is None:
                #data_format = "NWC"
                
            #elif data_format not in {"NCW", "NWC", "NCHW", "NHWC"}:
                #raise ValueError("data_format must be \"NWC\" or \"NCW\".")
            
            #self.strides = strides[0]
            #self.data_format = data_format
            #self.conv_op = self._conv1d
            
        elif conv_dims == 2:
            
            if data_format is None or data_format == "NHWC":
                data_format = "NHWC"
                strides = [1] + list(strides) + [1]
                
            elif data_format == "NCHW":
                ValueError('mpusim_slim._NonAtrousConvolution currently only supports NHWC format')
                #strides = [1, 1] + list(strides)
                
            else:
                raise ValueError("data_format must be \"NHWC\" or \"NCHW\".")
                self.strides = strides
                self.data_format = data_format
                
                self.conv_op = mpu_sim_conv2d_lib.mpusim_conv2d

        elif conv_dims == 3:
            raise ValueError('mpusim_slim._NonAtrousConvolution currently only supports 2D convolution')
            
            #if data_format is None or data_format == "NDHWC":
                #strides = [1] + list(strides) + [1]
                
            #elif data_format == "NCDHW":
                #strides = [1, 1] + list(strides)
                
            #else:
                #raise ValueError("data_format must be \"NDHWC\" or \"NCDHW\". Have: %s"
                                #% data_format)
            
            #self.strides = strides
            #self.data_format = data_format
            #self.conv_op = gen_nn_ops.conv3d

    def __call__(self, inp, filter):
        return self.conv_op(input=inp,
                                filter=filter,
                                strides=self.strides,
                                padding=self.padding,
                                data_format=self.data_format,
                                name=self.name)

class Convolution(object):

    def __init__(self,
                    input_shape,
                    filter_shape,
                    padding,
                    strides=None,
                    dilation_rate=None,
                    name=None,
                    data_format=None):
    
        num_total_dims = filter_shape.ndims
        
        if num_total_dims is None:
            num_total_dims = input_shape.ndims
            
        if num_total_dims is None:
            raise ValueError("rank of input or filter must be known")

            num_spatial_dims = num_total_dims - 2

        try:
            input_shape.with_rank(num_spatial_dims + 2)
            
        except ValueError:
            raise ValueError("input tensor must have rank %d" % (num_spatial_dims + 2))

        try:
            filter_shape.with_rank(num_spatial_dims + 2)
            
        except ValueError:
            raise ValueError( "filter tensor must have rank %d" % (num_spatial_dims + 2))

        if data_format is None or not data_format.startswith("NC"):
            input_channels_dim = tensor_shape.dimension_at_index(input_shape,
                                                                    num_spatial_dims + 1)
            spatial_dims = range(1, num_spatial_dims + 1)
            
        else:
            input_channels_dim = tensor_shape.dimension_at_index(input_shape, 1)
            spatial_dims = range(2, num_spatial_dims + 2)

        if not input_channels_dim.is_compatible_with(filter_shape[num_spatial_dims]):
            raise ValueError(
                "number of input channels does not match corresponding dimension of "
                "filter, {} != {}".format(input_channels_dim,
                                            filter_shape[num_spatial_dims]))

        strides, dilation_rate = nn_ops._get_strides_and_dilation_rate(num_spatial_dims,
                                                                                strides,
                                                                                dilation_rate)

        self.input_shape = input_shape
        self.filter_shape = filter_shape
        self.data_format = data_format
        self.strides = strides
        self.name = name
        self.conv_op = nn_ops._WithSpaceToBatch(input_shape,
                                                    dilation_rate=dilation_rate,
                                                    padding=padding,
                                                    build_op=self._build_op,
                                                    filter_shape=filter_shape,
                                                    spatial_dims=spatial_dims,
                                                    data_format=data_format)

    def _build_op(self, _, padding):
        
        # TODO: Change returned object type to custom _NonAtrousConvolution version
        return _NonAtrousConvolution(self.input_shape,
                                        filter_shape=self.filter_shape,
                                        padding=padding,
                                        data_format=self.data_format,
                                        strides=self.strides,
                                        name=self.name)

    def __call__(self, inp, filter):
        return self.conv_op(inp, filter)


class Conv(Layer):

    def __init__(self, rank,
                filters,
                kernel_size,
                strides=1,
                padding='valid',
                data_format=None,
                dilation_rate=1,
                activation=None,
                use_bias=True,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                trainable=True,
                name=None,
                **kwargs):
        super(Conv, self).__init__(trainable=trainable,
                                    name=name,
                                    activity_regularizer=regularizers.get(activity_regularizer),
                                    **kwargs)
        self.rank = rank
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(
            kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        if (self.padding == 'causal' and not isinstance(self,
                                                        (Conv1D, SeparableConv1D))):
            raise ValueError('Causal padding is only supported for `Conv1D`'
                            'and ``SeparableConv1D`.')
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(
            dilation_rate, rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=self.rank + 2)
        

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                            'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        if self.padding == 'causal':
            op_padding = 'valid'
        else:
            op_padding = self.padding
            
        # TODO: Change function call to custom Convolution function version
        self._convolution_op = nn_ops.Convolution(input_shape,
                                                    filter_shape=self.kernel.get_shape(),
                                                    dilation_rate=self.dilation_rate,
                                                    strides=self.strides,
                                                    padding=op_padding.upper(),
                                                    data_format=conv_utils.convert_data_format(self.data_format,
                                                                                                self.rank + 2))
        self.built = True
        

    def call(self, inputs):
        outputs = self._convolution_op(inputs, self.kernel)

        if self.use_bias:
            if self.data_format == 'channels_first':
            if self.rank == 1:
                # nn.bias_add does not accept a 1D input tensor.
                bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                outputs += bias
            if self.rank == 2:
                outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
            if self.rank == 3:
                # As of Mar 2017, direct addition is significantly slower than
                # bias_add when computing gradients. To use bias_add, we collapse Z
                # and Y into a single dimension to obtain a 4D input tensor.
                outputs_shape = outputs.shape.as_list()
                if outputs_shape[0] is None:
                    outputs_shape[0] = -1
                    outputs_4d = array_ops.reshape(outputs,
                                                    [outputs_shape[0], outputs_shape[1],
                                                    outputs_shape[2] * outputs_shape[3],
                                                    outputs_shape[4]])
                    outputs_4d = nn.bias_add(outputs_4d, self.bias, data_format='NCHW')
                    outputs = array_ops.reshape(outputs_4d, outputs_shape)
            else:
            outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

        if self.activation is not None:
            return self.activation(outputs)
        return outputs
    

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(space[i],
                                                            self.kernel_size[i],
                                                            padding=self.padding,
                                                            stride=self.strides[i],
                                                            dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return tensor_shape.TensorShape([input_shape[0]] + new_space + [self.filters])
        else:
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(space[i],
                                                            self.kernel_size[i],
                                                            padding=self.padding,
                                                            stride=self.strides[i],
                                                            dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return tensor_shape.TensorShape([input_shape[0], self.filters] + new_space)
        

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
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
        base_config = super(Conv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _compute_causal_padding(self):
        """Calculates padding for 'causal' option for 1-d conv layers."""
        left_pad = self.dilation_rate[0] * (self.kernel_size[0] - 1)
        if self.data_format == 'channels_last':
            causal_padding = [[0, 0], [left_pad, 0], [0, 0]]
        else:
            causal_padding = [[0, 0], [0, 0], [left_pad, 0]]
        return causal_padding


class Conv2D(Conv):
    def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1),
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
        super(Conv2D, self).__init__(rank=2,
                                        filters=filters,
                                        kernel_size=kernel_size,
                                        strides=strides,
                                        padding=padding,
                                        data_format=data_format,
                                        dilation_rate=dilation_rate,
                                        activation=activations.get(activation),
                                        use_bias=use_bias,
                                        kernel_initializer=initializers.get(kernel_initializer),
                                        bias_initializer=initializers.get(bias_initializer),
                                        kernel_regularizer=regularizers.get(kernel_regularizer),
                                        bias_regularizer=regularizers.get(bias_regularizer),
                                        activity_regularizer=regularizers.get(activity_regularizer),
                                        kernel_constraint=constraints.get(kernel_constraint),
                                        bias_constraint=constraints.get(bias_constraint),
                                        **kwargs)




