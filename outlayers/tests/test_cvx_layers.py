from outlayers.cvx_layers import CVXSoftmaxLayer
import unittest
import numpy as np
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
        assert_array_almost_equal(yc.detach().numpy(), yn.detach().numpy(), decimal=4)
        assert_array_almost_equal(xc.grad.detach().numpy(), xn.grad.detach().numpy(), decimal=4)

    def test_output(self):
        bs = 32
        size = 4
        softmax = CVXSoftmaxLayer(size)
        logits = torch.ones((bs, size), dtype=torch.float32)
        out = softmax(logits)
        assert_array_almost_equal(out.numpy(), 1/size*torch.ones((bs, size)).numpy())

    def test_loss(self):
        bs = 32
        size = 4
        softmax = CVXSoftmaxLayer(size)
        logits = torch.ones((bs, size), dtype=torch.float32)
        target = torch.ones((bs, 1), dtype=torch.long)
        target2 = torch.arange(0, bs, dtype=torch.long)[:, None] % size
        l = softmax.loss(logits, target)
        l2 = softmax.loss(logits, target2)
        self.assertEqual(l, l2)

    def test_get_targe_structure(self):
        size = 4
        softmax = CVXSoftmaxLayer(size)
        oe = softmax.get_output_encoding(size)
        max_targets, null_positions = oe.max_targets, oe.null_targets
        self.assertEqual(max_targets, [size-1])
        self.assertTrue(len(null_positions) == 1)
        self.assertTrue(null_positions[0] is None)

    def test_get_prediction(self):
        size = 4
        softmax = CVXSoftmaxLayer(size)
        score = np.array([[0.1, 0.2, 0.3, 0.4],
                          [0.05, 0.9, 0.05, 0]])
        pred = softmax.get_prediction(score)
        assert_array_almost_equal(pred, [[3], [1]])


if __name__ == '__main__':
    unittest.main()
