import abc
import torch
import torch.nn.functional as F


class AbstractOutLayer(abc.ABC):
    """Basic operations expected to be performed by the last layer."""

    @abc.abstractmethod
    def __call__(self, logit):
        """Compute `score` out of the `logits`.

        Here `logits` has shape (bs, logits_len) and can assume values in range (-Inf, Inf),
        `score` has the same shape (bs, logits_len) but can assume only values in (0, 1),
        and can be interpreted as probabilities.

        IMPORTANT: input and outputs are torch tensors!!
        """
        pass

    @abc.abstractmethod
    def loss(self, logits, target):
        """Compute `loss`: which measure the "discrepancy" between the model prediction and the observed target.

        Here `logits` has shape (bs, logits_len) and can assume values in range (-Inf, Inf),
        and `target` has shape (bs, target_len) and can assume only non-negative integer values.
        Check  `maximum_target` docstring for more information about the target values and shape.

        IMPORTANT: input and outputs are torch tensors!!
        """
        pass

    @abc.abstractmethod
    def get_target_structure(self, logits_len):
        """Returns the maximum value target is allowed to assume.

        The function signature is `maximum_target(logits_len: int) -> max_target, null_positions`.
        where both `max_target` and `null_positions` are list of the same length length,
        such that  `shape(target) = (bs, len(max_target)) = (bs, len(null_positions)) `

        Here the output `max_target` is such that  `0 <= target[:, i] <= max_target[i]` and
         .
        """
        pass

    @abc.abstractmethod
    def __repr__(self):
        return self._str()

    def __str__(self):
        return self._str()


class SoftmaxLayer(AbstractOutLayer):

    def __call__(self, logits):
        return F.softmax(logits, dim=-1)

    def loss(self, logits, target):
        score = F.log_softmax(logits, dim=-1)
        return F.nll_loss(score, target.flatten(), reduction='sum')

    def get_target_structure(self, logits_len):
        return [logits_len - 1], [None]

    def __repr__(self):
        return "standard_softmax"


class ReducedSoftmaxLayer(AbstractOutLayer):

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
        score = F.log_softmax(extended_logits, dim=-1)
        return F.nll_loss(score, target.flatten(), reduction='sum')

    def get_target_structure(self, logits_len):
        return [logits_len], [0]

    def __repr__(self):
        return "softmax"


class SigmoidLayer(AbstractOutLayer):

    def __call__(self, logits):
        return torch.sigmoid(logits)

    def loss(self, logits, target):
        score = self(logits)
        return F.binary_cross_entropy(score, target, reduction='sum')

    def get_target_structure(self, logits_len):
        return [1] * logits_len, [0] * logits_len

    def __repr__(self):
        return "sigmoid"


class ConcatenatedLayer(AbstractOutLayer):
    def __init__(self, layers, lengths):
        self.layers = layers
        self.lengths = lengths

    def _get_components(self, x, lengths):
        # Check lenght
        if x.size(-1) != sum(lengths):
            raise ValueError('Dimensions do not match')
        # Compute x
        components = []
        i = 0
        for l in range(lengths):
            components.append(x[..., i:i+l])
            i = i + l
        return components

    def __call__(self, logits):
        logits_components = self._get_components(logits, self.lengths)
        outputs = []
        for layer, logits in zip(self.layers, logits_components):
            outputs.append(layer(logits))
        return torch.cat(outputs, dim=-1)

    def loss(self, logits, targets):
        logits_components = self._get_components(logits, self.lengths)
        target_lengths = [len(layer.maximum_target(length)) for layer, length in zip(self.layers, self.lengths)]
        target_components = self._get_components(targets, target_lengths)

        loss = 0
        for layer, logits, targets in zip(self.layers, logits_components, target_components):
            loss += layer.loss(logits, targets)
        return loss

    def get_target_structure(self, logits_len):
        if logits_len != sum(self.lengths):
            raise ValueError('Invalid length')
        max_targets = []
        null_positions = []
        for layer, length in zip(self.layers, self.lengths):
            max_target, null_position = layer.maximum_target(length)
            max_targets += max_target
            null_positions += null_position
        return max_targets, null_position

    def __repr__(self):
        return 'concat:' + '-'.join(['{}:{}'.format(layer, length) for layer, length in zip(self.layers, self.lengths)])
