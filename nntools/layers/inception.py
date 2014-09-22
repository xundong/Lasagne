import numpy as np

import theano
import theano.tensor as T

from .. import init
from .. import nonlinearities
from .. import utils

from . import base
from . import cuda_convnet


def inception_module(l_in, num_1x1, reduce_3x3, num_3x3, reduce_5x5, num_5x5, num_pool_proj):
    shape = l_in.get_output_shape()
    out_layers = []

    # 1x1
    if num_1x1 > 0:
        l_1x1 = base.NINLayer(l_in, num_units=num_1x1)
        out_layers.append(l_1x1)
    
    # 3x3
    if num_3x3 > 0:
        if reduce_3x3 > 0:
            l_reduce_3x3 = base.NINLayer(l_in, num_units=reduce_3x3)
        else:
            l_reduce_3x3 = l_in
        l_3x3 = base.Conv2DLayer(l_reduce_3x3, num_filters=num_3x3, filter_size=(3, 3), border_mode="same")
        out_layers.append(l_3x3)
    
    # 5x5
    if num_5x5 > 0:
        if reduce_5x5 > 0:
            l_reduce_5x5 = base.NINLayer(l_in, num_units=reduce_5x5)
        else:
            l_reduce_5x5 = l_in
        l_5x5 = base.Conv2DLayer(l_reduce_5x5, num_filters=num_5x5, filter_size=(5, 5), border_mode="same")
        out_layers.append(l_5x5)
    
    # 3x3 pooling
    l_in_padded = base.PadLayer(l_in, width=1, batch_ndim=2)
    l_pool_3x3 = cuda_convnet.MaxPool2DCCLayer(l_in_padded, ds=(3, 3), strides=(1, 1))

    if num_pool_proj > 0:
        l_pool_proj = base.NINLayer(l_pool_3x3, num_units=num_pool_proj)
    else:
        l_pool_proj = l_pool_3x3
    out_layers.append(l_pool_proj)

    # stack
    l_out = base.ConcatLayer(out_layers)
    return l_out