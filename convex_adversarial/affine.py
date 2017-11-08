# Classes implementing Affine and AffineTranspose operators for pytorch layers
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

# Convert flattened input to convolutional input
def convert2to4(x, C, transpose=False): 
    if x.dim() == 1: 
        x = x.unsqueeze(0)
    m,n = x.size()
    channels = C.in_channels if not transpose else C.out_channels
    k = int((n//channels)**0.5)
    return x.contiguous().view(m,channels,k,k)

# Convert convolutional input to flattened input
def convert4to2(x): 
    m = x.size(0)
    return x.contiguous().view(m,-1)

class Affine(): 
    def __init__(self): 
        super(Affine, self).__init__()
        self.in_features = None
        self.out_features = None

    def __call__(self, x): 
        return self.forward(x)

    def record_size(self, x, y): 
        if x.dim() > 1: 
            self.in_features = x[0].numel()
            self.out_features = y[0].numel()
        else: 
            self.in_features = x.numel()
            self.out_features = y.numel()

class AffineLinear(Affine): 
    def __init__(self, l): 
        super(AffineLinear,self).__init__()
        if not isinstance(l, nn.Linear):
            raise ValueError("Expected nn.Linear input.")
        self.l = l

    def forward(self, x): 
        if x.dim() == 4: 
            x = convert4to2(x)
        out = F.linear(x, self.l.weight)
        self.record_size(x,out)
        return out

    def l1(self): 
        return self.l.weight.abs().sum(1)

class AffineConv2d(Affine): 
    def __init__(self, l): 
        super(AffineConv2d,self).__init__()
        if not isinstance(l, nn.Conv2d): 
            raise ValueError("Expected nn.Conv2d input.")
        self.l = l

    def forward(self, x): 
        if x.dim() == 2: 
            x = convert2to4(x, self.l)
        out = F.conv2d(x, self.l.weight, 
                       stride=self.l.stride,
                       padding=self.l.padding)
        out = convert4to2(out)
        self.record_size(x,out)
        return out

    def l1(self): 
        if self.out_features is None:
            raise ValueError("Affine layer not initialized.")
        oc, of = self.l.out_channels, self.out_features
        weight_sums = self.l.weight.view(self.l.out_channels, -1).abs().sum(1)
        return weight_sums[:,None].expand(oc,of//oc).contiguous().view(-1)

class AffineTransposeLinear(nn.Module): 
    def __init__(self, l): 
        super(AffineTransposeLinear, self).__init__()
        if not isinstance(l, nn.Linear):
            raise ValueError("Expected nn.Linear input.")
        self.l = l

    def forward(self, x): 
        if x.dim() == 4: 
            x = convert4to2(x)
        return F.linear(x, self.l.weight.t())

class AffineTransposeConv2d(nn.Module): 
    def __init__(self, l): 
        super(AffineTransposeConv2d, self).__init__()
        if not isinstance(l, nn.Conv2d): 
            raise ValueError("Expected nn.Conv2d input.")
        self.l = l

    def forward(self, x): 
        if x.dim() == 2: 
            x = convert2to4(x, self.l, transpose=True)
        out = F.conv_transpose2d(x, self.l.weight, 
                                 stride=self.l.stride,
                                 padding=self.l.padding)
        return convert4to2(out)
