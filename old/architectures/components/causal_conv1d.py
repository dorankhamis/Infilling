import torch.nn as nn

class CausalConv1d(nn.Conv1d):
    """ Taken from https://github.com/pytorch/pytorch/issues/1333 """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, x):
        result = super(CausalConv1d, self).forward(x)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result


class CausalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dilation=1, groups=1, bias=True, dropout=0.1):
        super(CausalBlock, self).__init__()            
        self.cconv1d = CausalConv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,            
            dilation=dilation,
            groups=groups,
            bias=bias)            
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.block = nn.Sequential(self.cconv1d, self.gelu, self.dropout)        

    def forward(self, x):
        return self.block(x)
