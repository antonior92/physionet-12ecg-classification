from .standard_layers import SigmoidLayer, SoftmaxLayer, ReducedSoftmaxLayer
from .cvx_layers import CVXSoftmaxLayer
from .concatenated_layers import ConcatenatedLayer
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
        sub_str = [s.split(':') for s in str.split("")[-1].split('-')]
        lengths = []
        layers = []
        for (name, lenght), l in zip(sub_str, self.layers):
            if not lenght.isnumeric():
                raise ValueError('Invalid output layer name')
            layers.append(from_str(name))
            lengths.append(int(lenght))
        return ConcatenatedLayer(layers, lengths)
    else:
        raise ValueError('Invalid output layer name')