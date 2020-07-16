from outlayers.cvx_layers import CVXSoftmaxLayer
import unittest
import torch
import torch.nn.functional as F
from numpy.testing import assert_array_almost_equal


class TestCVXSoftmax(unittest.TestCase):
    def test_softmax_sample(self):
        size = 10

        # Common softmax
        # instanciate
        torch.manual_seed(0)
        xn = torch.randn((size,), requires_grad=True)
        # forward pass
        yn = F.softmax(xn, dim=0)
        l = torch.sum((yn - 1)**2)
        # backward
        l.backward()

        # Convex softmax
        cvxsoftmax = CVXSoftmaxLayer(size)
        torch.manual_seed(0)
        xc = torch.randn((size,), requires_grad=True)
        # forward pass
        yc = cvxsoftmax(xc)
        l = torch.sum((yc - 1)**2)
        # backward
        l.backward()
        assert_array_almost_equal(yc.detach().numpy(), yn.detach().numpy())
        assert_array_almost_equal(xc.grad.detach().numpy(), xn.grad.detach().numpy())

    def test_softmax_batch(self):
        size = 10
        bs = 32

        # Common softmax
        # instanciate
        torch.manual_seed(0)
        xn = torch.randn((bs, size), requires_grad=True)
        # forward pass
        yn = F.softmax(xn, dim=1)
        l = torch.sum((yn - 1)**2)
        # backward
        l.backward()

        # Convex softmax
        cvxsoftmax = CVXSoftmaxLayer(size)
        torch.manual_seed(0)
        xc = torch.randn((bs, size,), requires_grad=True)
        # forward pass
        yc = cvxsoftmax(xc)
        l = torch.sum((yc - 1)**2)
        # backward
        l.backward()
        assert_array_almost_equal(yc.detach().numpy(), yn.detach().numpy())
        assert_array_almost_equal(xc.grad.detach().numpy(), xn.grad.detach().numpy())


if __name__ == '__main__':
    unittest.main()
