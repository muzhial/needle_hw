import numpy as np
import torch.nn as nn

import needle as ndl


def multi_node():
    v1 = ndl.Tensor([0], dtype="float32")
    v2 = ndl.exp(v1) + v1 + v1 ** 2

    print(len(v2.inputs))
    print(v2.op)
    print(v2.inputs[0])
    print(v2.inputs[1])

    print(v2.inputs[1].op)



def ex_matmul():
    a = np.random.randn(6, 6, 5, 4)
    b = np.random.randn(6, 6, 4, 3)

    c = np.random.randn(6, 6, 5, 3)
    d = np.random.randn(6, 6, 5)

    print(np.matmul(a, b).shape)  # (6, 6, 5, 3)
    print(np.matmul(d, c).shape)


if __name__ == '__main__':
    ex_matmul()
