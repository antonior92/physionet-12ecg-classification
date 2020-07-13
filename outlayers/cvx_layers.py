import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch.nn.functional as F
import torch
import numpy as np


class CVXSoftmaxLayer(object):

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

    def __call__(self, x):
        y, = self._softmax(x)

        return y

    def loss(self, logits, target):
        score = torch.log(self(logits))

        return F.nll_loss(score, target, reduction='sum')



