import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


#################
### for model ###
#################
kVeryBigNumber = 1e30
kVerySmallNumber = 1e-30
kVeryPositiveNumber = kVeryBigNumber
kVeryNegativeNumber = -kVeryBigNumber

# assuming val is cuda FloatTensor
def zero_mask(val, mask):
	return val * mask.float()

def exp_mask(val, mask):
	return val + (1 - mask.float()) * kVeryNegativeNumber


################
### for data ###
################
""" Read all (train and test combined) data, return lists of (passage, question, answer) tuples and word dictionaries."""
def read_data(filename):
	



################
### datasets ###
################


