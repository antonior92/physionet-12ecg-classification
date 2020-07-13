import torch
import torch.nn.functional as F


class SoftmaxLayer(object):

    def __init__(self, dtype=torch.float32):
        self.dtype = dtype

    def __call__(self, logits):
        return F.softmax(logits, dim=-1, dtype=self.dtype)

    def loss(self, logits, target):
        score = self(logits)
        return F.nll_loss(score, target, reduction='sum')


class SigmoidLayer(object):

    def __init__(self, dtype=torch.float32):
        self.dtype = dtype

    def __call__(self, logits):
        return torch.sigmoid(logits)

    def loss(self, logits, target):
        score = self(logits)

        return F.binary_cross_entropy(score, target, reduction='sum')