import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch.nn.functional as F
from .abstract_out_layer import AbstractOutLayer
import torch
import numpy as np


class CVXSoftmaxLayer(abc.ABC):

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
        return F.nll_loss(score, target, reduction='sum')

    def maximum_target(self, logits_len):
        return [logits_len - 1]

    def get_prediction(self, score):
        return score.argmax(axis=-1)

    def __str__(self):
        return "cvx_softmax_{}".format(self.size)

    def __repr__(self):
        return "cvx_softmax_{}".format(self.size)

    @classmethod
    def from_str(cls, str):
        if "cvx_softmax_" in str and str.split("cvx_softmax_")[-1].isnumeric():
            size = str.split("cvx_softmax_")[-1]
            return cls(size)
        else:
            raise ValueError('Unknown string')


