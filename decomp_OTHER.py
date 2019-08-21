import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
from datetime import timedelta
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import tensorly as tl
import numpy as np
import argparse

# added on march 27
from collections import OrderedDict

from TVBMF_OTHER import EVBMF
from torch_cp_decomp_OTHER import torch_cp_decomp
from torch_tucker_OTHER import tucker_decomp



def tucker_rank(layer):
    W = layer.weight.data
    mode3 = tl.base.unfold(W, 0)
    mode4 = tl.base.unfold(W, 1)
    diag_0 = EVBMF(mode3)
    diag_1 = EVBMF(mode4)
    d1 = diag_0.shape[0]
    d2 = diag_1.shape[1]

    del mode3
    del mode4
    del diag_0
    del diag_1

    # round to multiples of 16
    return [int(np.ceil(d1 / 16) * 16) \
            , int(np.ceil(d2 / 16) * 16)]


def est_rank(layer):
    W = layer.weight.data
    mode3 = tl.base.unfold(W, 0)
    mode4 = tl.base.unfold(W, 1)
    diag_0 = EVBMF(mode3)
    diag_1 = EVBMF(mode4)

    # round to multiples of 16
    return int(np.ceil(max([diag_0.shape[0], diag_1.shape[0]]) / 16) * 16)


if __name__ == '__main__':
    main()
