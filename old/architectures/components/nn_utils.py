import numpy as np
import torch
import torch.nn as nn
import copy

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
    
def alltrue_mask(size):
    "Mask template"
    attn_shape = (1, size, size)
    no_mask = np.zeros(attn_shape, dtype='uint8')
    return torch.from_numpy(no_mask) == 0
    
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])  

def conv_len_out_size(len_in, kern, pad, dil, stride):
    return 1 + (len_in + 2*pad - dil * (kern-1) - 1) / stride

def determine_conv_layers(seq_len, kern, pad, dil, stride):
    s = seq_len
    inc = 0
    while s > 3:
        try:
            s = conv_len_out_size(s, kern, pad, dil, stride)
            inc += 1
        except:
            break
    return int(s), inc
