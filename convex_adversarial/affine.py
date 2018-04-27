# Classes implementing Affine and AffineTranspose operators for pytorch layers
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from .dense import Dense

# Conversion functions to turn layers into affine layers
def toAffineTranspose(l): 
    if isinstance(l, nn.Linear): 
        return AffineTransposeLinear(l)
    elif isinstance(l, nn.Conv2d): 
        return AffineTransposeConv2d(l)
    elif isinstance(l, Dense): 
        raise ValueError('Cannot transpose Dense layers by themselves.')
        # return AffineTransposeDense(l)
    elif isinstance(l, nn.Sequential) and len(l) == 0:
        return nn.Sequential() 
    else:
        raise ValueError("AffineTranspose class not found for given layer.")

def toAffine(l): 
    if isinstance(l, nn.Linear): 
        return AffineLinear(l)
    elif isinstance(l, nn.Conv2d): 
        return AffineConv2d(l)
    elif isinstance(l, Dense): 
        return AffineDense(l)
    elif isinstance(l, nn.Sequential) and len(l) == 0:
        return nn.Sequential() 
    else:
        raise ValueError("Affine class not found for given layer.")

# helper function to extract biases from layers
def full_bias(l, n=None): 
    # expands the bias to the proper size. For convolutional layers, a full
    # output dimension of n must be specified. 
    if isinstance(l, nn.Linear): 
        return l.bias.view(1,-1)
    elif isinstance(l, nn.Conv2d): 
        if n is None: 
            raise ValueError("Need to pass n=<output dimension>")
        b = l.bias.unsqueeze(1).unsqueeze(2)
        k = int((n/(b.numel()))**0.5)
        return b.expand(b.numel(),k,k).contiguous().view(1,-1)
    elif isinstance(l, Dense): 
        return sum(full_bias(layer, n=n) for layer in l.Ws)
    elif isinstance(l, nn.Sequential) and len(l) == 0: 
        return 0
    else:
        raise ValueError("Full bias can't be formed for given layer.")

# Convolutional helper functions to minibatch large inputs for CuDNN
def conv2d(x, *args, **kwargs): 
    """ Minibatched inputs to conv2d """
    i = 0
    out = []
    batch_size = 10000
    while i < x.size(0): 
        out.append(F.conv2d(x[i:min(i+batch_size, x.size(0))], *args, **kwargs))
        i += batch_size
    return torch.cat(out, 0)

def conv_transpose2d(x, *args, **kwargs):
    i = 0
    out = []
    batch_size = 10000
    while i < x.size(0): 
        out.append(F.conv_transpose2d(x[i:min(i+batch_size, x.size(0))], *args, **kwargs))
        i += batch_size
    return torch.cat(out, 0)

# Convert flattened input to convolutional input
def convert2to4(x, C, transpose=False): 
    if x.dim() == 1: 
        x = x.unsqueeze(0)
    m,n = x.size()
    channels = C.in_channels if not transpose else C.out_channels
    k = int((n//channels)**0.5)
    return x.view(m,channels,k,k)

# Convert convolutional input to flattened input
def convert4to2(x): 
    m = x.size(0)
    return x.view(m,-1)

# Base Affine class
def num_features(x): 
    if x.dim() > 1: 
        return x[0].numel()
    else:
        return x.numel()

class Affine(): 
    def __init__(self): 
        # super(Affine, self).__init__()
        self.in_features = None
        self.out_features = None

    def record_size(self, x, y): 
        if isinstance(x, list): 
            self.in_features = [num_features(x0) for x0 in x]
        else: 
            self.in_features = num_features(x)
        self.out_features = num_features(y)

class AffineLinear(nn.Module, Affine): 

    def __init__(self, l): 
        super(AffineLinear,self).__init__()
        Affine.__init__(self)
        if not isinstance(l, nn.Linear):
            raise ValueError("Expected nn.Linear input.")
        self.l = l

    def forward(self, *xs): 
        x = xs[-1]
        if x.dim() == 4: 
            x = convert4to2(x)
        out = F.linear(x, self.l.weight)

        if self.in_features is None or self.out_features is None: 
            self.record_size(x,out)

        return out

class AffineConv2d(nn.Module, Affine): 

    def __init__(self, l): 
        super(AffineConv2d,self).__init__()
        Affine.__init__(self)
        if not isinstance(l, nn.Conv2d): 
            raise ValueError("Expected nn.Conv2d input.")
        self.l = l

    def forward(self, *xs): 
        x = xs[-1]
        if x.dim() == 2: 
            x = convert2to4(x, self.l)
        out = conv2d(x, self.l.weight, 
                       stride=self.l.stride,
                       padding=self.l.padding)
        out = convert4to2(out)

        if self.in_features is None or self.out_features is None: 
            self.record_size(x,out)

        return out

class AffineTransposeLinear(nn.Module): 

    def __init__(self, l): 
        super(AffineTransposeLinear, self).__init__()
        if not isinstance(l, nn.Linear):
            raise ValueError("Expected nn.Linear input.")
        self.l = l

    def forward(self, *xs): 
        x = xs[-1]
        if x.dim() == 4: 
            x = convert4to2(x)
        return F.linear(x, self.l.weight.t())

class AffineTransposeConv2d(nn.Module): 
    
    def __init__(self, l): 
        super(AffineTransposeConv2d, self).__init__()
        if not isinstance(l, nn.Conv2d): 
            raise ValueError("Expected nn.Conv2d input.")
        self.l = l

    def forward(self, *xs): 
        x = xs[-1]
        if x.dim() == 2: 
            x = convert2to4(x, self.l, transpose=True)
        out = conv_transpose2d(x, self.l.weight, 
                                 stride=self.l.stride,
                                 padding=self.l.padding)
        return convert4to2(out)

class AffineDense(Dense, Affine): 
    def __init__(self, D): 
        super(AffineDense, self).__init__(*list(toAffine(W) for W in D.Ws))
        Affine.__init__(self)

    def forward(self, *xs): 
        out = super(AffineDense, self).forward(*xs)        
        if self.in_features is None or self.out_features is None: 
            self.in_features = [num_features(x) for x in xs]
            self.out_features = num_features(out)
        return out


def transpose_all(ls): 
    if all(isinstance(l, Dense) for l in ls):
        layers = [Dense() for l in ls]
        for i,l in reversed(list(enumerate(ls))): 
            for j,W in enumerate(l.Ws): 
                layers[i+j-len(l.Ws)+1].Ws.append(toAffineTranspose(W))
        return layers
        raise NotImplementedError
    elif all(not isinstance(l,Dense) for l in ls):
        return [toAffineTranspose(l) for l in ls]
    else:
        raise ValueError('In order to convert Dense Affine layers we ' 
                         'need all layers to be dense in this '
                         'implementation. ')

# class AffineTransposeDense(Dense): 
#     def __init__(self, ls): 