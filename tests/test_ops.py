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
