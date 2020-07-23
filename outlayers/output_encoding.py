from copy import copy

class OutputEncoding():
    """Define correpondence between score positions and the target positions and values."""
    def __init__(self, max_targets, null_targets):
        # Define one-to-one correpondence between:
        # (idx, subidx) <-> indice
        # Such that `score[:, indice]` correspond to `target[idx] == subidx`
        i = 0
        indices = []
        pairs = []
        for idx, (max_target, null_target) in enumerate(zip(max_targets, null_targets)):
            for subidx in range(max_target+1):
                if subidx != null_target:
                    pairs.append((idx, subidx))
                    indices.append(i)
                    i += 1

        self.max_targets = max_targets
        self.null_targets = null_targets
        self.dict_from_indice_to_pair = dict(zip(indices, pairs))
        self.dict_from_pair_to_indice = dict(zip(pairs, indices))
        self._pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def is_null(self, pair):
        idx, subidx = pair
        return subidx == self.null_targets[idx]

    def get_no_null_subidx(self, idx):
        m = self.max_targets[idx]
        n = self.null_targets[idx]
        return [i for i in range(m+1) if i != n]

    @property
    def pairs(self):
        return copy(self._pairs)
