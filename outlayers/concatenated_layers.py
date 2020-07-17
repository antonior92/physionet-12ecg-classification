import torch
from .abstract_out_layer import AbstractOutLayer


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

    def maximum_targets(self, logits_len):
        if logits_len != sum(self.lengths):
            raise ValueError('Invalid length')
        list_lengts = []
        for layer, length in zip(self.layers, self.lengths):
            list_lengts += layer.maximum_target(length)
        return list_lengts

    def get_prediction(self, score):
        score_components = self._get_components(score, self.lengths)
        pred = []
        for layer, score_i in zip(self.layers, score_components):
            pred.append(layer.get_predictions(score_i))
        return torch.cat(pred, dim=-1)

    def __str__(self):
        return 'concat:' + '-'.join(['{}:{}'.format(layer, length) for layer, length in zip(self.layers, self.lengths)])

    def __repr__(self):
        return 'concat:' + '-'.join(['{}:{}'.format(layer, length) for layer, length in zip(self.layers, self.lengths)])

