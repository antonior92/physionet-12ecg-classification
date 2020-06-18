import torch
import numpy as np


def get_collapse_fun(args):
    # choose collapse function
    if args['pred_stage_type'].lower() == 'mean':
        return lambda y: np.mean(y, axis=0)
    elif args['pred_stage_type'].lower() == 'max':
        return lambda y: np.max(y, axis=0)
    elif args['pred_stage_type'].lower() in ['lstm', 'gru', 'rnn']:
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
    def __init__(self, bs, softmax_mask, device, dtype=torch.float32):
        # Save zero tensor to be used as 'normal' case in the softmax
        self.normal = torch.zeros((bs, 1), device=device, dtype=dtype)
        self.softmax_mask = softmax_mask
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='sum')
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='sum')
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=-1)

    def _get_output_components(self, output):
        bs = output.size(0)
        softmax_outputs = []
        n_softmax_outputs = 0
        for mask in self.softmax_mask:
            softmax_outputs.append(torch.cat((self.normal[:bs], output[:, mask]), dim=1))
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
    CLASSES = ['AF', 'I-AVB', 'RBBB', 'LBBB', 'PAC', 'PVC', 'STD', 'STE']
    IDX = [0, 0, 1, 1, 2, 3, 4, 5]

    @property
    def uniquedict(self):
        idx = self.IDX
        unique_idx = np.unique(idx)
        m = dict(zip(unique_idx, [[] for _ in range(len(unique_idx))]))
        for i, id in enumerate(idx):
            m[id].append(i)
        return m

    @property
    def mutually_exclusive(self):
        m = self.uniquedict
        return [l for l in list(m.values()) if len(l) > 1]

    @property
    def subidx(self):
        m = self.uniquedict
        subidx = []
        for l in m.values():
            for i in range(len(l)):
                subidx.append(i)
        return subidx

    @property
    def class_to_idx(self):
        return dict(zip(self.CLASSES, self.IDX))

    @property
    def class_to_subidx(self):
        return dict(zip(self.CLASSES, self.subidx))

    def __len__(self):
        return len(self.CLASSES)

    @property
    def len_target(self):
        return len(self) - len(self.mutually_exclusive)

    def multiclass_to_binaryclass(self, x):
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

    def add_normal_column(self, x, prob=False):
        x = np.atleast_2d(x)
        n_samples, n_classes = x.shape
        new_x = np.zeros((n_samples, n_classes + 1), dtype=x.dtype)
        new_x[:, :-1] = x[:, :]
        # If x is a vector of zeros and ones
        if not prob:
            new_x[:, -1] = x.sum(axis=1) == 0
        # if x is a vector of probabilities
        else:
            counter = 0
            new_x[:, -1] = 1.0
            for mask in self.mutually_exclusive:
                new_x[:, -1] = x[:, -1] * (1 - x[:, mask].sum(axis=1))
                counter += len(mask)
            x[:, -1] = x[:, -1] * np.prod(1 - x[:, counter:], axis=1)

        return np.squeeze(new_x)

    def get_target_from_labels(self, labels):
        target = np.zeros(self.len_target)
        for l in labels:
            if l in self.CLASSES:
                target[self.class_to_idx[l]] = self.class_to_subidx[l] + 1
        return target






