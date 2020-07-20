import numpy as np

# TODO: merge get_item + prepare_probabilities
# TODO: implement get_prediction
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
        new_prob = np.zeros(list(prob.shape[:-1]) + [len(classes)], dtype=prob.dtype)
        for i, c in enumerate(classes):
            if c in self.classes:
                new_prob[..., i] = layer.get_item(prob, self.class_to_idx[c], self.class_to_subidx[l])
        return new_prob


def get_item(score, idx, subidx, target_structure):
    """Returns the `score` (i.e. probability) which is correspondes to the hypotesis that `target[:, idx] = subidx`

    Be idx/subidx list of values. `len(idx) === len(subidx) == item_len` Here `score` has shape (bs, logits_len)
    and the returned value has shape (bs, item_len).

    IMPORTANT: input and outputs are numpy arrays!!
    """
    if isinstance(subidx, int):
        subidx = [subidx]

    if self.NULL_TARGET in subidx:
        null_target = self.NULL_TARGET
        subidx_no_null_target = list(subidx)
        subidx_no_null_target.remove(self.NULL_TARGET)
    else:
        subidx_no_null_target = subidx
        null_target = None
    subidx_no_null_target = [i - (1 if i > self.NULL_TARGET else 0) for i in subidx_no_null_target]
    if null_target is not None:
        bs, score_len = score.shape
        items = np.zeros((bs, len(subidx)), dtype=score.dtype)
        # Get subdix without null target
        # Asign values
        if len(subidx_no_null_target) >= 1:
            items[:, np.array(subidx) != self.NULL_TARGET] = score[:, subidx_no_null_target]
        items[:, null_target] = 1 - score.sum(axis=1)
        return items
    else:
        return score[:, subidx_no_null_target]



def get_item(self, score, _, subidx):
    if isinstance(subidx, int):
        return score[:, [subidx]]
    else:
        return score[:, subidx]


def get_item(self, score, idx, subidx):
    bs, logits_len = score.shape
    max_targets, which_sublayer = self._maximum_targets(logits_len)
    which_sublayer = np.array(which_sublayer)  # to facilitate manipulation
    idx_sublayer = idx - min(np.where(which_sublayer == which_sublayer[idx])[0])
    item = self.layers[which_sublayer].get_item(score[:, index_in_sublayer], idx_sublayer, subidx)
    return item



def get_item(self, score, idx, subidx):
    if subidx != self.NULL_TARGET:
        return score[:, idx]
    else:
        return 1 - score.sum(axis=1)




def get_prediction(self, score):
    score_components = self._get_components(score, self.lengths)
    pred = []
    for layer, score_i in zip(self.layers, score_components):
        pred.append(layer.get_predictions(score_i))
    return torch.cat(pred, dim=-1)



def get_prediction(score, target_structure):
    """ Get most likely prediction out of the score function.

    Here `score` has the same shape (bs, logits_len) but can assume only values in (0, 1),
    and can be interpreted as probabilities. The output, prediction, on the other hand, has
    shape (bs, target_len). The prediction is an object similar to `target`, but predicted
    by the model (rather than observed in the dataset).

    IMPORTANT: input and outputs are numpy arrays!!
    """
    complete_score = np.hstack([1 - np.sum(score, axis=1, keepdims=True), score])
    return complete_score.argmax(axis=-1)[:, None]



def get_prediction(self, score):
    return score.argmax(axis=-1)


def get_prediction(self, score):
    return score.argmax(axis=-1)[:, None]


def get_prediction(self, score):
    return score > 0.5

def get_prediction(score, target_structure):
    """ Get most likely prediction out of the score function.

    Here `score` has the same shape (bs, logits_len) but can assume only values in (0, 1),
    and can be interpreted as probabilities. The output, prediction, on the other hand, has
    shape (bs, target_len). The prediction is an object similar to `target`, but predicted
    by the model (rather than observed in the dataset).

    IMPORTANT: input and outputs are numpy arrays!!
    """
    complete_score = np.hstack([1 - np.sum(score, axis=1, keepdims=True), score])
    return complete_score.argmax(axis=-1)[:, None]


# Softmax

def test_get_prediction(self):
    score = np.array([[0.1, 0.2, 0.3, 0.4],
                      [0.05, 0.9, 0.05, 0]])
    pred = self.softmax.get_prediction(score)
    assert_array_almost_equal(pred, [[3], [1]])

def test_get_items_1(self):
    score = np.array([[0.1, 0.2, 0.3, 0.4],
                      [0.05, 0.9, 0.05, 0]])
    item = self.softmax.get_item(score, [0, 0], [0, 1])
    assert_array_almost_equal(item, [[0.1, 0.2], [0.05, 0.9]])

def test_get_items_2(self):
    score = np.array([[0.1, 0.2, 0.3, 0.4],
                       [0.05, 0.9, 0.05, 0]])
    item = self.softmax.get_item(score, 0, 1)
    assert_array_almost_equal(item, [[0.2], [0.9]])

# Reduced softmax

def test_get_prediction(self):
    score = np.array([[0.2, 0.3, 0.4],
                      [0.9, 0.05, 0],
                      [0.1, 0.05, 0]])
    pred = self.softmax.get_prediction(score)
    assert_array_almost_equal(pred, [[3], [1], [0]])

def test_get_items_1(self):
    score = np.array([[0.2, 0.3, 0.4],
                      [0.9, 0.05, 0],
                      [0.1, 0.05, 0]])
    item = self.softmax.get_item(score, [0, 1], [0, 1])
    assert_array_almost_equal(item, [[0.1, 0.2], [0.05, 0.9], [0.85, 0.1]])

def test_get_items_2(self):
    score = np.array([[0.2, 0.3, 0.4],
                      [0.9, 0.05, 0],
                      [0.1, 0.05, 0]])
    item = self.softmax.get_item(score, 0, 0)
    assert_array_almost_equal(item, [[0.1], [0.05], [0.85]])

def test_get_items_3(self):
    score = np.array([[0.2, 0.3, 0.4],
                      [0.9, 0.05, 0],
                      [0.1, 0.05, 0]])
    item = self.softmax.get_item(score, [1, 2], [1, 2])
    assert_array_almost_equal(item, [[0.2, 0.3], [0.9, 0.05], [0.1, 0.05]])