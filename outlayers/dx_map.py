import numpy as np

class DxMap(object):

    def __init__(self, classes, idx, subidx):
        self.class_to_idx = dict(zip(classes, idx))
        self.class_to_subidx = dict(zip(classes, subidx))
        self.classes = classes
        self.target_len = max(idx)

    def target_from_labels(self):
        len_target = self.target_len
        target = np.zeros(len_target, dtype=int)

        for l in labels:
            if l in self.classes:
                target[self.class_to_idx[l]] = self.class_to_subidx[l]
        return target

    def prepare_target(self, target, classes):
        new_target = np.zeros(list(target.shape[:-1]) + [len(classes)], dtype=y.dtype)
        for i, c in enumerate(classes):
            if c in self.classes:
                new_target[..., i] = target[self.class_to_idx[c]] == self.class_to_subidx[l]
        return new_target

    def prepare_probabilities(self, prob, classes, layer):
        new_prob = np.zeros(list(prob.shape[:-1]) + [len(classes)], dtype=y.dtype)
        for i, c in enumerate(classes):
            if c in self.classes:
                new_prob[..., i] = layer.get_item(prob, self.class_to_idx[c], self.class_to_subidx[l])
        return new_prob