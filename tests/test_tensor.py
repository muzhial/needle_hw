import unittest

import needle as ndl


class TestTensor(unittest.TestCase):

    def test_tensor_init(self):
        x = ndl.Tensor([1, 2, 3], dtype='float32')
        self.assertTrue(isinstance(x, ndl.Value))
        self.assertTrue(x.requires_grad)
        self.assertTrue(len(x.inputs) == 0)
        self.assertTrue(x.dtype == 'float32')
        self.assertTrue(x.op is None)
        self.assertTrue(x.device == ndl.cpu())
        self.assertEqual(x.shape, (3,))
