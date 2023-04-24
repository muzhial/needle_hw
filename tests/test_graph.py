import numpy as np

import needle as ndl


def test_compute_graph_1():
        v1 = ndl.Tensor([0], dtype="float32")
        v2 = ndl.exp(v1)
        v3 = v2 + 1
        v4 = v2 * v3

        assert(
            np.all(v4.cached_data == np.array([2.0], dtype=np.float32))
        )

        assert v4.inputs[0] is v2
        assert v4.inputs[1] is v3

        assert v3.op.__dict__ == {'scalar': 1}
