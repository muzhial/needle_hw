import unittest

import numpy as np
import needle as ndl


class TestOps(unittest.TestCase):

    def test_add_scalar(self):
        x = ndl.Tensor([1, 2, 3])
        y = ndl.add_scalar(x, 1)
        self.assertTrue(isinstance(y, ndl.Value))
        self.assertTrue(y.requires_grad)
        self.assertTrue(len(y.inputs) == 1)
        self.assertTrue(y.inputs[0] is x)
        self.assertTrue(isinstance(y.op, ndl.AddScalar))
        self.assertTrue(np.all(y.cached_data == np.array([2, 3, 4])))

    def test_power_scalar(self):
        x = ndl.Tensor([1, 2, 3])
        y = ndl.power_scalar(x, 2)
        self.assertTrue(np.all(y.cached_data == np.array([1, 4, 9])))

        x = ndl.Tensor([3.0], dtype="float32")
        y = x ** 2
        self.assertTrue(
            np.all(y.cached_data == np.array([9.0], dtype=np.float32)))

        with self.assertRaises(TypeError):
            x = ndl.Tensor([3.0], dtype="float32")
            y = x ** "2"

        with self.assertRaises(TypeError):
            x = ndl.Tensor([3.0], dtype="float32")
            ep = ndl.Tensor([2])
            y = x ** ep

    def test_divide_scalar(self):
        x = ndl.Tensor([1, 2, 3])
        y = x / 2
        self.assertTrue(
            np.all(y.cached_data == np.array([0.5, 1.0, 1.5])))

        y1 = ndl.divide_scalar(x, 2)
        self.assertTrue(
            np.all(y1.cached_data == np.array([0.5, 1.0, 1.5]))
        )

        # with self.assertRaises(ZeroDivisionError,):
        #     y1 = x / 0

    def test_divide(self):
        x1 = ndl.Tensor([3, 9])
        x2 = ndl.Tensor([2])
        y1 = x1 / x2
        y2 = ndl.divide(x1, x2)
        self.assertTrue(
            np.all(y1.cached_data == np.array([1.5, 4.5]))
        )
        self.assertTrue(
            np.all(y1.cached_data == y2.cached_data)
        )

    def test_sub(self):
        x1 = ndl.Tensor([1, 2, 3])
        x2 = ndl.Tensor([1, 1, 1])
        x3 = ndl.Tensor([1])
        x4 = 1
        y1 = x1 - x2
        self.assertTrue(
            np.all(y1.cached_data == np.array([0, 1, 2]))
        )
        y2 = x1 - x3
        self.assertTrue(
            np.all(y2.cached_data == np.array([0, 1, 2]))
        )
        y3 = x1 - x4
        self.assertTrue(
            np.all(y3.cached_data == np.array([0, 1, 2]))
        )

    def test_matmul(self):
        x1 = ndl.Tensor([[1, 0],
                         [0, 1]])
        x2 = ndl.Tensor([[4, 1],
                         [2, 2]])
        y1 = x1 @ x2
        self.assertTrue(
            np.all(y1.cached_data == x2.cached_data)
        )
        y2 = ndl.matmul(x1, x2)
        self.assertTrue(
            np.all(y2.cached_data == x2.cached_data)
        )

    def test_reshape(self):
        x1 = ndl.Tensor([1, 2, 3, 4, 5, 6])
        y1 = x1.reshape((2, 3))
        self.assertTrue(y1.shape == (2, 3))

    def test_transpose(self):
        x1 = ndl.Tensor([[1, 2, 3,],
                         [4, 5, 6],
                         [7, 8, 9]])
        y1 = x1.transpose((1, 0))
        self.assertTrue(y1.shape == (3, 3))
        self.assertTrue(
            np.all(y1.cached_data == np.array([[1, 4, 7],
                                               [2, 5, 8],
                                               [3, 6, 9]]))
        )

    def test_log(self):
        x1 = ndl.Tensor([1, 2, 3])
        y1 = ndl.log(x1)
        self.assertTrue(
            np.allclose(
                y1.cached_data,
                np.array([0, 0.69314718, 1.09861229]),
                atol=1e-5)
        )
