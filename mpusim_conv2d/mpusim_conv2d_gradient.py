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

from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops

from tensorpack.compat import tfv1 as tf  # this should be avoided first in model code

from tensorpack.tfutils.common import get_tf_version_tuple
from tensorpack.utils.argtools import get_data_format, shape2d, shape4d, log_once
from tensorpack.models.common import VariableHolder, layer_register
from tensorpack.models.tflayer import convert_to_tflayer_args, rename_get_variable

@ops.RegisterGradient("MpuSimConv2D")
def _MpuSimConv2DGrad(op, grad):
  """Gradient function for MpuSimConv2D."""
  dilations = op.get_attr("dilations")
  strides = op.get_attr("strides")
  padding = op.get_attr("padding")
  data_format = op.get_attr("data_format")
  shape_0, shape_1 = array_ops.shape_n([op.inputs[0], op.inputs[1]])

  # We call the gen_nn_ops backprop functions instead of nn_ops backprop
  # functions for performance reasons in Eager mode. gen_nn_ops functions take a
  # `explicit_paddings` parameter, but nn_ops functions do not. So if were were
  # to use the nn_ops functions, we would have to convert `padding` and
  # `explicit_paddings` into a single `padding` parameter, increasing overhead
  # in Eager mode.
  return [
      nn_ops.conv2d_backprop_input(
          shape_0,
          op.inputs[1],
          grad,
          dilations=dilations,
          strides=strides,
          padding=padding,
          use_cudnn_on_gpu=False,
          data_format=data_format),
      nn_ops.conv2d_backprop_filter(
          op.inputs[0],
          shape_1,
          grad,
          dilations=dilations,
          strides=strides,
          padding=padding,
          use_cudnn_on_gpu=False,
          data_format=data_format)
  ]
