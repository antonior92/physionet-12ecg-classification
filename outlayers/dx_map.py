import numpy as np


class DxMap(object):
    # TODO: include testing

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

    def prepare_target(self, target, classes):
        new_target = np.zeros(list(target.shape[:-1]) + [len(classes)], dtype=target.dtype)
        for i, c in enumerate(classes):
            if c in self.classes:
                # convert to list
                c2p = self.class_to_pair[c] if isinstance(self.class_to_pair[c], list) else [self.class_to_pair[c]]
                # assign target from labels
                idx, subidx = c2p[0]
                new_target[..., i] = target[idx] == subidx
                for idx, subidx in c2p[1:]:
                    new_target[..., i] *= target[idx] == subidx
        return new_target

    def prepare_probabilities(self, prob, classes):
        bs, score_len = prob.shape
        new_prob = np.zeros((bs, len(classes)), dtype=prob.dtype)
        for i, c in enumerate(classes):
            if c in self.classes:
                c2p = self.class_to_pair[c] if isinstance(self.class_to_pair[c], list) else [self.class_to_pair[c]]

                pair = c2p[0]
                if self.enc.is_null(pair):
                    index = self.enc.dict_from_pair_to_indice[pair]
                    new_prob[:, i] = prob[:, index]
                else:
                    indices = self.enc.get_no_null_subidx(pair[0])
                    new_prob[:, i] = prob[:, indices].sum(axis=-1)

                for pair in c2p[1:]:
                    if self.enc.is_null(pair):
                        index = self.enc.dict_from_pair_to_indice[pair]
                        new_prob[:, i] *= prob[:, index]
                    else:
                        indices = self.enc.get_no_null_subidx(pair[0])
                        new_prob[:, i] *= 1 - prob[:, indices].sum(axis=-1)
        return new_prob

    def __len__(self):
        return len(self.enc)
