import numpy as np


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
                index = [classes.index(a) for a in alias_group]
                index.sort()
                p = pairs[index[0]]
                for k in index[1:]:
                    pairs.insert(k, p)
        return cls(classes, pairs, enc)

    def __init__(self, classes, pairs, enc):
        self.class_to_pair = dict(zip(classes, pairs))
        self.classes = classes
        self.enc = enc

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

    def prepare_target(self, target, classes=None):
        target = np.array(target, dtype=int)
        if classes is None:
            indices = list(range(len(self.enc)))
            pairs_list = [[p] for p in self.enc.pairs]
        else:
            indices, pairs_list = self._get_pairs(classes)

        new_target = np.zeros(list(target.shape[:-1]) + [len(pairs_list)], dtype=target.dtype)
        for i, pairs in zip(indices, pairs_list):
            # assign target from labels
            idx, subidx = pairs[0]
            new_target[..., i] = target[..., idx] == subidx
            for idx, subidx in pairs[1:]:
                new_target[..., i] *= target[..., idx] == subidx
        return new_target

    def prepare_probabilities(self, prob, classes):
        prob = np.array(prob, dtype=float)
        bs, score_len = prob.shape
        indices, pairs_list = self._get_pairs(classes)
        new_prob = np.zeros((bs, len(pairs_list)), dtype=prob.dtype)
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
