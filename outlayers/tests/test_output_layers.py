import unittest
import torch
import numpy as np
from outlayers import SoftmaxLayer, ReducedSoftmaxLayer, SigmoidLayer
from numpy.testing import assert_array_almost_equal


class TestSoftmax(unittest.TestCase):
    def setUp(self):
        self.softmax = SoftmaxLayer()

    def test_output(self):
        bs = 32
        size = 4
        logits = torch.ones((bs, size), dtype=torch.float32)
        out = self.softmax(logits)
        assert_array_almost_equal(out.numpy(), 1/size*torch.ones((bs, size)).numpy())

    def test_loss(self):
        bs = 32
        size = 4
        logits = torch.ones((bs, size), dtype=torch.float32)
        target = torch.ones((bs, 1), dtype=torch.long)
        target2 = torch.arange(0, bs, dtype=torch.long)[:, None] % size
        l = self.softmax.loss(logits, target)
        l2 = self.softmax.loss(logits, target2)
        self.assertEqual(l, l2)

    def test_get_targe_structure(self):
        oe = self.softmax.get_output_encoding(10)
        max_targets, null_positions = oe.max_targets, oe.null_targets
        self.assertEqual(max_targets, [9])
        self.assertTrue(len(null_positions) == 1)
        self.assertTrue(null_positions[0] is None)

    def test_get_prediction(self):
        score = np.array([[0.1, 0.2, 0.3, 0.4],
                          [0.05, 0.9, 0.05, 0]])
        pred = self.softmax.get_prediction(score)
        assert_array_almost_equal(pred, [[3], [1]])


class TestReducedSoftmax(unittest.TestCase):
    def setUp(self):
        self.softmax = ReducedSoftmaxLayer()

    def test_output(self):
        bs = 32
        size = 4
        logits = torch.zeros((bs, size-1), dtype=torch.float32)
        out = self.softmax(logits)
        assert_array_almost_equal(out.numpy(), 1/size*torch.ones((bs, size-1)).numpy())

    def test_output2(self):
        bs = 32
        size = 4
        logits = torch.ones((bs, size-1), dtype=torch.float32)
        out = self.softmax(logits)
        out2 = SoftmaxLayer()(torch.cat([torch.zeros((bs, 1)), logits], dim=1))
        assert_array_almost_equal(out.numpy(), out2[:, 1:].numpy())

    def test_loss(self):
        bs = 32
        size = 4
        logits = torch.zeros((bs, size), dtype=torch.float32)
        target = torch.ones((bs, 1), dtype=torch.long)
        target2 = torch.arange(0, bs, dtype=torch.long)[:, None] % size
        l = self.softmax.loss(logits, target)
        l2 = self.softmax.loss(logits, target2)
        self.assertEqual(l, l2)

    def test_get_targe_structure(self):
        oe = self.softmax.get_output_encoding(10)
        max_targets, null_positions = oe.max_targets, oe.null_targets
        self.assertEqual(max_targets, [10])
        self.assertTrue(null_positions, [0])

    def test_get_prediction(self):
        score = np.array([[0.2, 0.3, 0.4],
                          [0.9, 0.05, 0],
                          [0.1, 0.05, 0]])
        pred = self.softmax.get_prediction(score)
        assert_array_almost_equal(pred, [[3], [1], [0]])


class TestSigmoid(unittest.TestCase):
    def setUp(self):
        self.sigmoid = SigmoidLayer()

    def test_output(self):
        bs = 32
        size = 4
        logits = torch.zeros((bs, size), dtype=torch.float32)
        out = self.sigmoid(logits)
        assert_array_almost_equal(out.numpy(), 1/2*torch.ones((bs, size)).numpy())

    def test_loss(self):
        bs = 32
        size = 4
        logits = torch.zeros((bs, size), dtype=torch.float32)
        target = torch.cat([torch.ones((bs, 1)), torch.zeros((bs, size-1))], dim=1)
        target2 = torch.cat([torch.zeros((bs, size-1)), torch.ones((bs, 1))], dim=1)
        l = self.sigmoid.loss(logits, target)
        l2 = self.sigmoid.loss(logits, target2)
        self.assertEqual(l, l2)

    def test_get_targe_structure(self):
        oe = self.sigmoid.get_output_encoding(10)
        max_targets, null_positions = oe.max_targets, oe.null_targets
        self.assertEqual(max_targets, [1]*10)
        self.assertTrue(null_positions, [0]*10)

    def test_get_prediction(self):
        score = np.array([[0.2, 0.3, 0.4],
                          [0.9, 0.05, 0],
                          [0.1, 0.05, 0]])
        pred = self.sigmoid.get_prediction(score)
        assert_array_almost_equal(pred, [[0, 0, 0],
                                         [1, 0, 0],
                                         [0, 0, 0]])

if __name__ == '__main__':
    unittest.main()
