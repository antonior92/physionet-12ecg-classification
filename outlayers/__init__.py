from .output_layers import SigmoidLayer, SoftmaxLayer, ReducedSoftmaxLayer, ConcatenatedLayer
try:
    from .cvx_layers import CVXSoftmaxLayer
except:
    pass
from .collapse import collapse, get_collapse_fun
from .dx_map import DxMap

__all__ = ['DxMap', 'collapse', 'get_collapse_fun', 'outlayer_from_str']


def outlayer_from_str(str):
    if str == 'softmax':
        return ReducedSoftmaxLayer()
    elif str == 'sigmoid':
        return SigmoidLayer()
    elif str == 'standard_softmax':
        return SoftmaxLayer()
    elif "cvx_softmax_" in str and str.split("cvx_softmax_")[-1].isnumeric():
        size = str.split("cvx_softmax_")[-1]
        return CVXSoftmaxLayer(size)
    elif 'concat:' in str:
        str = str.split("concat:")[-1]
        sub_str = [s.split(':') for s in str.split("concat:")[-1].split('-')]
        layers, lengths = zip(*[(outlayer_from_str(n), int(l)) for n, l in sub_str])
        return ConcatenatedLayer(layers, lengths)
    else:
        raise ValueError('Invalid output layer name {:}'.format(str))