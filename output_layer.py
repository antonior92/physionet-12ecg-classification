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
