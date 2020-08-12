import numpy as np
import warnings


def get_collapse_fun(tp):
    # choose collapse function
    if tp == 'mean':
        return lambda y: np.mean(y, axis=0)
    elif tp == 'max':
        return lambda y: np.max(y, axis=0)
    elif tp == 'last':
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

