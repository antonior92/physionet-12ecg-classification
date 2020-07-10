import torch
import operator
import functools
import numpy as np
import pandas as pd


def get_collapse_fun(tp):
    # choose collapse function
    if tp == 'mean':
        return lambda y: np.mean(y, axis=0)
    elif tp == 'max':
        return lambda y: np.max(y, axis=0)
    elif tp in ['lstm', 'gru', 'rnn']:
        return lambda y: y[-1]


def collapse(x, ids, fn, unique_ids=None):
    """Collapse arrays with the same ids using fn.

    Be `x` an array (N, *) and ids a sequence with N elements, possibly with repeated entries, `M` unique ids
    return a tuple containing the unique ids and a array with shape (M, *)  where the i-th entry
    is obtaining by applying fn to all entries in `x` with the same id.

    fn should be a function that colapse the first dimention of the array: `fn: ndarray shape(N, *) -> (*)`
    """
    ids = np.array(ids)
    # Get unique ids
    if unique_ids is None:
        unique_ids = np.unique(ids)
    # Collapse using fn
    new_x = np.zeros((len(unique_ids), *x.shape[1:]), dtype=x.dtype)
    for i, id in enumerate(unique_ids):
        new_x[i, ...] = fn(x[ids == id, ...])

    return unique_ids, new_x


class OutputLayer(object):
    def __init__(self, bs, dx, device, dtype=torch.float32):
        self.softmax_mask = dx.mutually_exclusive
        if self.softmax_mask:  # if not empty
            # Save zero tensor to be used as 'null_class' case in the softmax
            self.null_class = torch.zeros((bs, 1), device=device, dtype=dtype)
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='sum')
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='sum')
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=-1)

    def _get_output_components(self, output):
        bs = output.size(0)
        softmax_outputs = []
        n_softmax_outputs = 0
        for mask in self.softmax_mask:
            softmax_outputs.append(torch.cat((self.null_class[:bs], output[:, mask]), dim=1))
            n_softmax_outputs += len(mask)
        sigmoid_outputs = output[:, n_softmax_outputs:]
        return softmax_outputs, sigmoid_outputs

    def _get_target_components(self, target):
        softmax_targets = []
        i = 0
        for _ in self.softmax_mask:
            softmax_targets.append(target[:, i])
            i += 1
        sigmoid_targets = target[:, i:]
        return softmax_targets, sigmoid_targets

    def loss(self, output, target):
        softmax_outputs, sigmoid_outputs = self._get_output_components(output)
        softmax_targets, sigmoid_targets = self._get_target_components(target)

        loss = 0
        for i, _ in enumerate(self.softmax_mask):
            loss += self.ce_loss(softmax_outputs[i], softmax_targets[i].to(dtype=torch.long))
        loss += self.bce_loss(sigmoid_outputs, sigmoid_targets.to(dtype=torch.float32))

        return loss

    def get_output(self, output):
        softmax_outputs, sigmoid_outputs = self._get_output_components(output)

        outputs = []
        for i, mask in enumerate(self.softmax_mask):
            outputs.append(self.softmax(softmax_outputs[i])[..., -len(mask):])
        outputs.append(self.sigmoid(sigmoid_outputs))

        return torch.cat(outputs, dim=-1)


class DxClasses(object):

    def __init__(self, classes, mutually_exclusive=None, null_class=None):
        # Get idx and subidx
        idx = []
        code = []
        subidx = []
        i = 0
        # Convert to set/string
        classes = set([str(c) for c in classes])
        # Remove mutually_exclussive from classes
        if mutually_exclusive is not None:
            all_mutually_exclusive = functools.reduce(operator.add, (m for m in mutually_exclusive))
            classes = classes.difference(all_mutually_exclusive)
            for mutually_exclusive_sublist in mutually_exclusive:
                j = 0
                for c in mutually_exclusive_sublist:
                    # set idx and group
                    idx.append(i)
                    code.append(c)
                    subidx.append(j)
                    j += 1
                i += 1
        if null_class is not None:
            classes = classes.difference([null_class])

        # Convert to list
        classes = list(classes)
        for c in classes:
            # set idx and group
            idx.append(i)
            code.append(c)
            subidx.append(0)
            i += 1

        self.classes = classes
        self._mutually_exclusive = mutually_exclusive
        self.null_class = null_class
        self.idx = idx
        self.code = code
        self.subidx = subidx
        self.target_len = i

    def _null_column(self, x, prob=False):
        x = np.atleast_2d(x)
        n_samples, n_classes = x.shape
        # If x is a vector of zeros and ones
        if not prob:
            return x.sum(axis=1) == 0
        # if x is a vector of probabilities
        else:
            counter = 0
            null_column = np.ones(n_samples, dtype=x.dtype)
            for mask in self.mutually_exclusive:
                null_column = null_column * (1 - x[:, mask].sum(axis=1))
                counter += len(mask)
            null_column = null_column * np.prod(1 - x[:, counter:], axis=1)
            return null_column

    @property
    def mutually_exclusive(self):
        m = []
        if self._mutually_exclusive is not None:
            for mutually_exclusive_sublist in self._mutually_exclusive:
                aux_list = []
                for c in mutually_exclusive_sublist:
                    aux_list.append(self.code.index(c))
                m.append(aux_list)
        return m

    def __len__(self):
        return len(self.code)

    def get_target_from_labels(self, labels):
        len_target = self.target_len
        target = np.zeros(len_target)

        class_to_idx = dict(zip(self.code, self.idx))
        class_to_subidx = dict(zip(self.code, self.subidx))
        for l in labels:
            if l in self.code:
                target[class_to_idx[l]] = class_to_subidx[l] + 1
        return target

    def target_to_binaryclass(self, x):
        x = np.atleast_2d(x)
        n_samples = x.shape[0]
        new_x = np.zeros((n_samples, len(self)), dtype=bool)

        counter = 0
        for i, mask in enumerate(self.mutually_exclusive):
            for j, id in enumerate(mask):
                new_x[:, id] = x[:, i] == (j + 1)
                counter += 1
        new_x[:, counter:] = x[:, len(self.mutually_exclusive):]
        return np.squeeze(new_x)

    def reorganize(self, y, classes, prob=False):
        dict_current_order = dict(zip(self.code, range(len(self.code))))
        new_idx = [dict_current_order[c] for c in classes if c in self.code]
        valid_idx = np.isin(classes, self.code)
        new_y = np.zeros(list(y.shape[:-1]) + [len(classes)], dtype=y.dtype)
        new_y[..., valid_idx] = y[..., new_idx]
        if (self.null_class is not None) and (self.null_class in classes):
            new_y[..., classes.index(self.null_class)] = self._null_column(y, prob)
        return new_y

    def compute_threshold(self, dset, ids):
        dset.use_only_header(True)
        targets = np.stack([self.get_target_from_labels(s['labels']) for s in dset[ids]])
        dset.use_only_header(False)
        y_true = self.target_to_binaryclass(targets)
        threshold = y_true.sum(axis=0) / y_true.shape[0]
        return threshold

    def apply_threshold(self, y_score, threshold):
        n_records = y_score.shape[0]
        n_softmax_outputs = 0
        y_pred = np.zeros((n_records, self.target_len), dtype=int)
        for i, mask in enumerate(self.mutually_exclusive):
            # Compute masked value
            y_masked = y_score[:, mask]
            y_subgroup = np.hstack([1 - np.sum(y_masked, axis=1)[:, None], y_masked])
            # Correct for threshold
            threshold_masked = threshold[mask]
            threshold_subgroup = np.hstack([1 - np.sum(threshold_masked), threshold_masked])
            y_subgroup = y_subgroup / threshold_subgroup
            # Compute maximum value
            y_subgroup.argmax(axis=1, out=y_pred[:, i])
            # Go to next one
            n_softmax_outputs += len(mask)
        y_pred[:, len(self.mutually_exclusive):] = y_score[:, n_softmax_outputs:] > threshold[n_softmax_outputs:]
        return y_pred
