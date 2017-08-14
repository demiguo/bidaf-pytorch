import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


kVeryBigNumber = 1e30
kVerySmallNumber = 1e-30
kVeryPositiveNumber = kVeryBigNumber
kVeryNegativeNumber = -kVeryBigNumber

# assuming val is cuda FloatTensor
def zero_mask(val, mask):
	return val * mask.float()

def exp_mask(val, mask):
	return val + (1 - mask.float()) * kVeryNegativeNumber
