import torch
import torch.nn.functional as F
from .abstract_out_layer import AbstractOutLayer


class SoftmaxLayer(AbstractOutLayer):

    STR = "standard_softmax"

    def __call__(self, logits):
        return F.softmax(logits, dim=-1)

    def loss(self, logits, target):
        score = F.log_softmax(logits)
        return F.nll_loss(score, target.flatten(), reduction='sum')

    def maximum_target(self, logits_len):
        return [logits_len - 1]

    def get_prediction(self, score):
        return score.argmax(axis=-1)


    def __str__(self):
        return self.STR

    def __repr__(self):
        return self.STR



class ReducedSoftmaxLayer(AbstractOutLayer):

    STR = "softmax"

    def _add_zero(self, logits):
        return torch.cat((torch.zeros(logits.size(0), 1), logits), dim=1)

    def _remove_extra_column(self, output):
        return output[..., 1:]

    def __call__(self, logits):
        extended_logits = self._add_zero(logits)
        output = F.softmax(extended_logits, dim=-1)
        return self._remove_extra_column(output)

    def loss(self, logits, target):
        extended_logits = self._add_zero(logits)
        score = F.log_softmax(extended_logits)
        return F.nll_loss(score, target.flatten(), reduction='sum')

    def maximum_target(self, logits_len):
        return [logits_len]

    def get_prediction(self, score):
        complete_score = np.hstack([1 - np.sum(score, axis=1, keepdims=True), score])
        return complete_score.argmax(axis=-1)

    def __str__(self):
        return self.STR

    def __repr__(self):
        return self.STR

    @classmethod
    def from_str(cls, str):
        if str == self.STR:
            return cls()
        else:
            raise ValueError('Unknown string')



class SigmoidLayer(AbstractOutLayer):

    STR = "sigmoid"

    def __call__(self, logits):
        return torch.sigmoid(logits)

    def loss(self, logits, target):
        score = self(logits)
        return F.binary_cross_entropy(score, target, reduction='sum')

    def maximum_target(self, logits_len):
        return [1] * logits_len

    def get_prediction(self, score):
        return score > 0.5

    def __str__(self):
        return "sigmoid"

    def __repr__(self):
        return "sigmoid"

    @classmethod
    def (cls, str):
        if str == 'softmax':
            return cls()
        else:
            raise ValueError('Unknown string')