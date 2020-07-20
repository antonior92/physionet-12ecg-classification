import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch.nn.functional as F
from .output_layers import AbstractOutLayer
import torch
import numpy as np


class CVXSoftmaxLayer(AbstractOutLayer):

    EPS = 1e-4

    def __init__(self, size, dtype=np.float32):
        # Define softmax layer
        x = cp.Parameter(size)
        y = cp.Variable(size)
        obj = cp.Minimize(-x.T @ y - cp.sum(cp.entr(y)))
        cons = [np.ones(size, dtype=dtype).T @ y == 1.,
                y >= self.EPS,
                y <= 1]
        prob = cp.Problem(obj, cons)
        self._softmax = CvxpyLayer(prob, parameters=[x], variables=[y])
        self.size = size

    def __call__(self, x):
        y, = self._softmax(x)
        return y

    def loss(self, logits, target):
        score = torch.log(self(logits))
        return F.nll_loss(score, target.flatten(), reduction='sum')

    def get_target_structure(self, logits_len):
        return [logits_len - 1], [None]

    def __str__(self):
        return "cvx_softmax_{}".format(self.size)

    def __repr__(self):
        return "cvx_softmax_{}".format(self.size)


