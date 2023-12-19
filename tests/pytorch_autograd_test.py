import torch
import torch.nn as nn
import torch.nn.functional as F


def into_grad():
    v1 = torch.randn(3, 3, 4, requires_grad=True)
    v3 = torch.empty_like(v1)
    v2 = v1.int()
    v2.zero_()
    print(v1, '\n', v2)
    print(v1.requires_grad, v2.requires_grad, v3.requires_grad)

    t1 = v1.view(-1, 4)
    print(t1.requires_grad)

    print('=' * 29)

    x = torch.ones(2, 2, requires_grad=True)
    print(x.grad, x.grad_fn)
    y = x + 2
    m = y * y * 2
    z = m.mean()
    print(y.grad, y.grad_fn)
    print(z.grad, z.grad_fn)
    # y.zero_()  # error
    # m.zero_()  # no error
    # z.zero_()  # no error
    with torch.no_grad():
        p_y = x * x * 2
        print(f'p_y, {p_y.requires_grad}')

    z.backward()
    # x.zero_()  # RuntimeError: a leaf Variable that requires grad has been used in an in-place operation
    # y.zero_()  # no error why?
    print('=== after backward')
    print(x.grad, x.grad_fn)
    print(y.grad, y.grad_fn)
    print(z.grad, z.grad_fn)


def tensor_storage():
    v = torch.tensor([1, 2, 3])
    vi = v.clone()
    vii = v.reshape((1, 3))
    vi.fill_(0)
    print(v, v.storage().data_ptr())
    print(vi, vi.storage().data_ptr())
    print(vii, vii.storage().data_ptr())

    a = torch.randn(2, 2, 2, 2)
    print(a)
    # a[0, [0, 1, 1, 0], 0:2, 0:2] = 0      # right
    # a[0, [0, 1, 1, 0], [0, 0], [0, 1]] = 0  # error
    a[0, [0, 1, 1, 0], [0, 0, 1, 1], [0, 1, 0, 1]] = 0
    print(a)


class X:

    def __init__(self):
        super(X, self).__init__()
        self.x = torch.ones((2, 2), requires_grad=True)

    def forward(self):
        y = self.x + 3
        z = y * y * 3
        out = z.mean(dim=1)

        return out

x = X()
for _ in range(3):
    print(x.x)
    out = x.forward()
    print(out)
#     x.x.zero_grad()
    out.backward(torch.tensor([1.0, 1.0]))
    print(x.x.grad)


#######
A = torch.tensor([1.0, 2.0], requires_grad=True)
B = torch.tensor([3.0, 5.0], requires_grad=True)
C = torch.tensor([4.0, 7.0], requires_grad=True)

D = A * B
E = A + B
F = C + B
G = D + (E + F)

G_fn = G.grad_fn
print(G)
print(G_fn)
print(G_fn.next_functions)
print(G_fn(torch.tensor(1)))


#######
A = torch.tensor([3.], requires_grad=True)
B = torch.tensor([2.], requires_grad=True)
C = A ** 2  # 9
D = B ** 2  # 4
E = D * C   # 36
F = D + E   # 40

F.manual_grad = torch.tensor(1)
D.manual_grad, E.manual_grad = F.grad_fn(F.manual_grad)
print(D.manual_grad, E.manual_grad)
tmp1, C.manual_grad = E.grad_fn(E.manual_grad)
D.manual_grad = D.manual_grad + tmp1
A.manual_grad = C.grad_fn(C.manual_grad)
B.manual_grad = D.grad_fn(D.manual_grad)
