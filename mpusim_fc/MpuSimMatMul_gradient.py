# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python import pywrap_tensorflow as c_api
from tensorflow.python.compat import compat
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops

@ops.RegisterGradient("MpuSimMatMul")
def _MpuSimMatMulGrad(op, grad):
    """Gradient for MatMul."""
    try:
        skip_input_indices = op.skip_input_indices
        if skip_input_indices is not None:
            if 1 in skip_input_indices:
                return _MatMulGradAgainstFirstOnly(op, grad)
            elif 0 in skip_input_indices:
                return _MatMulGradAgainstSecondOnly(op, grad)
    except AttributeError:
        # No gradient skipping, so do the full gradient computation
        pass

    t_a = op.get_attr("transpose_a")
    t_b = op.get_attr("transpose_b")
    a = math_ops.conj(op.inputs[0])
    b = math_ops.conj(op.inputs[1])
    if not t_a and not t_b:
        grad_a = gen_math_ops.mat_mul(grad, b, transpose_b=True)
        grad_b = gen_math_ops.mat_mul(a, grad, transpose_a=True)
    elif not t_a and t_b:
        grad_a = gen_math_ops.mat_mul(grad, b)
        grad_b = gen_math_ops.mat_mul(grad, a, transpose_a=True)
    elif t_a and not t_b:
        grad_a = gen_math_ops.mat_mul(b, grad, transpose_b=True)
        grad_b = gen_math_ops.mat_mul(a, grad)
    elif t_a and t_b:
        grad_a = gen_math_ops.mat_mul(b, grad, transpose_a=True, transpose_b=True)
        grad_b = gen_math_ops.mat_mul(grad, a, transpose_a=True, transpose_b=True)
    return grad_a, grad_b

