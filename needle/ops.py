"""Operator and gradient implementations."""
from numbers import Number
from typing import List, Optional

# NOTE: we will numpy as the array_api
# to backup our computations, this line will change in later homeworks
import numpy
import numpy as array_api

from .autograd import NDArray, Op, Tensor, TensorOp, Value, cpu


class EWiseAdd(TensorOp):

    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):

    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):

    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):

    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar, )


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return array_api.power(a, self.scalar)

    def gradient(self, out_grad, node):
        in_val = node.inputs[0]
        return (self.scalar * array_api.power(in_val, self.scalar - 1) *
                out_grad, )


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return out_grad / rhs, -out_grad * lhs / rhs**2


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):

    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad, node):
        return (out_grad / self.scalar, )


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):

    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        # TODO: to optimize
        if self.axes is None:
            return array_api.swapaxes(a, -1, -2)
        if len(self.axes) == 2:
            return array_api.swapaxes(a, self.axes[0], self.axes[1])
        return array_api.transpose(a, self.axes)

    def gradient(self, out_grad, node):
        return transpose(out_grad, axes=self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):

    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad, node):
        in_shape = node.inputs[0].shape
        return reshape(out_grad, in_shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):

    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        in_shape = node.inputs[0].shape
        sum_axes = []
        in_pad_shape = [None
                        ] * (len(self.shape) - len(in_shape)) + list(in_shape)
        for i, (in_dim, out_dim) in enumerate(zip(in_pad_shape, self.shape)):
            if in_dim is None or in_dim != out_dim:
                sum_axes.append(i)
        sum_axes = tuple(sum_axes)
        return reshape(summation(out_grad, axes=sum_axes), in_shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):

    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.sum(a, axis=self.axes)

    def gradient(self, out_grad, node):
        in_shape = node.inputs[0].shape
        grad_shape = list(in_shape)
        if self.axes is None:
            grad_shape[:] = [1] * len(grad_shape)
        if isinstance(self.axes, int):
            grad_shape[self.axes] = 1
        if isinstance(self.axes, (tuple, list)):
            for axis in self.axes:
                grad_shape[axis] = 1
        return broadcast_to(reshape(out_grad, grad_shape), in_shape)


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):

    def compute(self, a, b):
        return array_api.matmul(a, b)

    def gradient(self, out_grad, node):
        """
        d(XY) = (dX)Y + X(dY)
        """
        lhs, rhs = node.inputs
        lhs_grad, rhs_grad = (matmul(out_grad, transpose(rhs)),
                              matmul(transpose(lhs), out_grad))
        if lhs_grad.shape != lhs.shape:
            lhs_grad = summation(
                lhs_grad,
                axes=tuple(range(len(lhs_grad.shape) - len(lhs.shape))))
        if rhs_grad.shape != rhs.shape:
            rhs_grad = summation(
                rhs_grad,
                axes=tuple(range(len(rhs_grad.shape) - len(rhs.shape))))
        return lhs_grad, rhs_grad


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):

    def compute(self, a):
        return -1 * a

    def gradient(self, out_grad, node):
        return -1 * out_grad


def negate(a):
    return Negate()(a)


class Log(TensorOp):

    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        return out_grad / node.inputs[0]


def log(a):
    return Log()(a)


class Exp(TensorOp):

    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        return out_grad * exp(node.inputs[0])


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):

    def compute(self, a):
        array_api.maximum(a, 0)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class LogSoftmax(TensorOp):

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


# additional helper functions
def full(shape,
         fill_value,
         *,
         rand={},
         dtype='float32',
         device=None,
         requires_grad=False):
    # numpy do not need device argument
    kwargs = {'device': device} if array_api is not numpy else {}
    device = device if device else cpu()

    if not rand or 'dist' not in rand:
        arr = array_api.full(shape, fill_value, dtype=dtype, **kwargs)
    else:
        if rand['dist'] == 'normal':
            arr = array_api.randn(shape,
                                  dtype,
                                  mean=rand['mean'],
                                  std=rand['std'],
                                  **kwargs)
        if rand['dist'] == 'binomial':
            arr = array_api.randb(shape,
                                  dtype,
                                  ntrials=rand['trials'],
                                  p=rand['prob'],
                                  **kwargs)
        if rand['dist'] == 'uniform':
            arr = array_api.randu(shape,
                                  dtype,
                                  low=rand['low'],
                                  high=rand['high'],
                                  **kwargs)

    return Tensor.make_const(arr, requires_grad=requires_grad)


def zeros(shape, *, dtype='float32', device=None, requires_grad=False):
    return full(shape,
                0,
                dtype=dtype,
                device=device,
                requires_grad=requires_grad)


def randn(shape,
          *,
          mean=0.0,
          std=1.0,
          dtype='float32',
          device=None,
          requires_grad=False):
    return full(
        shape,
        0,
        rand={
            'dist': 'normal',
            'mean': mean,
            'std': std
        },
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


def randb(shape,
          *,
          n=1,
          p=0.5,
          dtype='float32',
          device=None,
          requires_grad=False):
    return full(
        shape,
        0,
        rand={
            'dist': 'binomial',
            'trials': n,
            'prob': p
        },
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


def randu(shape,
          *,
          low=0,
          high=1,
          dtype='float32',
          device=None,
          requires_grad=False):
    return full(
        shape,
        0,
        rand={
            'dist': 'uniform',
            'low': low,
            'high': high
        },
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


def zeros_like(array, *, device=None, requires_grad=False):
    device = device if device else array.device
    return full(array.shape,
                0,
                dtype=array.dtype,
                device=device,
                requires_grad=requires_grad)


def ones_like(array, *, device=None, requires_grad=False):
    device = device if device else array.device
    return full(array.shape,
                1,
                dtype=array.dtype,
                device=device,
                requires_grad=requires_grad)
