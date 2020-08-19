import numpy as np
from .output_encoding import OutputEncoding


class DxMap(object):
    """Map between classes and output layer."""

    @classmethod
    def infer_from_out_layer(cls, classes, out_layer, alias=None):
        # Get all idx subidx for training classes
        outlayer_len = len(classes) - (sum([len(a) - 1 for a in alias]) if alias is not None else 0)
        enc = out_layer.get_output_encoding(outlayer_len)
        pairs = enc.pairs
        # Deal with alias groups
        if alias is not None:
            for alias_group in alias:
                index = [classes.index(a) for a in alias_group if a in classes]
                index.sort()
                p = pairs[index[0]]
                for k in index[1:]:
                    pairs.insert(k, p)
        return cls(classes, pairs, enc)

    @classmethod
    def from_str(cls, string):
        lines = string.split('\n')
        encoding_aux = lines[0]
        classes_aux, idx_aux, subidx_aux = zip(*[line.split(',') for line in lines[1:]])
        idx_aux, subidx_aux = [int(i) for i in idx_aux], [int(j) for j in subidx_aux]  # to int
        unique_classes = list(set(classes_aux))

        pairs_list = []
        classes = []
        for c in unique_classes:
            classes.append(c)
            pairs = []
            for l, i, j in zip(classes_aux, idx_aux, subidx_aux):
                if l == c:
                    pairs.append((i, j))
            pairs_list.append(pairs)
        return cls(classes, pairs_list, OutputEncoding.from_str(encoding_aux))

    def __init__(self, classes, pairs_list, enc):
        self.class_to_pair = dict(zip(classes, pairs_list))
        self.classes = classes
        self.enc = enc

    @property
    def classes_at_the_output(self):
        """Classes available at the neural network output.

        The string '---' is return """
        classes = []
        pairs_list = self.enc.pairs
        pair_to_class = {}
        for cls, pairs in self.class_to_pair.items():
            if isinstance(pairs, list):
                for pair in pairs:
                    pair_to_class[pair] = cls
            else:
                pair = pairs
                pair_to_class[pair] = cls
        for pair in pairs_list:
            try:
                classes.append(pair_to_class[pair])
            except:
                classes.append('---')
        return classes

    def target_from_labels(self, labels):
        len_target = len(self.enc.max_targets)
        target = np.zeros(len_target, dtype=int)

        for l in labels:
            if l in self.classes:
                # convert to list
                c2p = self.class_to_pair[l] if isinstance(self.class_to_pair[l], list) else [self.class_to_pair[l]]
                # assign target from labels
                for idx, subidx in c2p:
                    target[idx] = subidx
        return target

    def _get_pairs(self, classes):
        pairs_list = []
        indices = []
        for i, c in enumerate(classes):
            if c in self.classes:
                pairs = self.class_to_pair[c] if isinstance(self.class_to_pair[c], list) else[self.class_to_pair[c]]
                pairs_list.append(pairs)
                indices.append(i)
        return indices, pairs_list

    def _prepare_target(self, target, indices, pairs_list, n_classes=None):
        if n_classes is None:
            n_classes = len(indices)
        new_target = np.zeros(list(target.shape[:-1]) + [n_classes], dtype=target.dtype)
        for i, pairs in zip(indices, pairs_list):
            # assign target from labels
            idx, subidx = pairs[0]
            new_target[..., i] = target[..., idx] == subidx
            for idx, subidx in pairs[1:]:
                new_target[..., i] *= target[..., idx] == subidx
        return new_target

    def prepare_target(self, target, classes=None):
        target = np.array(target, dtype=int)
        if classes is None:
            indices = list(range(len(self.enc)))
            pairs_list = [[p] for p in self.enc.pairs]
        else:
            indices, pairs_list = self._get_pairs(classes)
        n_classes = len(classes) if classes is not None else len(indices)
        return self._prepare_target(target, indices, pairs_list, n_classes)

    def prepare_probabilities(self, prob, classes):
        prob = np.array(prob, dtype=float)
        bs, score_len = prob.shape
        indices, pairs_list = self._get_pairs(classes)
        n_classes = len(classes) if classes is not None else len(indices)
        new_prob = np.zeros((bs, n_classes), dtype=prob.dtype)
        for i, pairs in zip(indices, pairs_list):
            pair = pairs[0]
            if not self.enc.is_null(pair):
                index = self.enc.dict_from_pair_to_indice[pair]
                new_prob[:, i] = prob[:, index]
            else:
                idx, _ = pair
                subidx = self.enc.get_no_null_subidx(idx)
                indices = [self.enc.dict_from_pair_to_indice[(idx, si)] for si in subidx]
                new_prob[:, i] = 1 - prob[:, indices].sum(axis=-1)
            for pair in pairs[1:]:
                if not self.enc.is_null(pair):
                    index = self.enc.dict_from_pair_to_indice[pair]
                    new_prob[:, i] *= prob[:, index]
                else:
                    idx, _ = pair
                    subidx = self.enc.get_no_null_subidx(idx)
                    indices = [self.enc.dict_from_pair_to_indice[(idx, si)] for si in subidx]
                    new_prob[:, i] *= 1 - prob[:, indices].sum(axis=-1)
        return new_prob

    def __len__(self):
        return len(self.enc)

    def __repr__(self):
        repr = [str(self.enc)]
        _, pairs_list = self._get_pairs(self.classes)
        for c, pairs in zip(self.classes, pairs_list):
            for idx, subidx in pairs:
                repr.append('{:},{:},{:}'.format(c, idx, subidx))
        return '\n'.join(repr)

    def __str__(self):
        return self.__repr__()
