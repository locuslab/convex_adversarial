# Classes implementing Affine and AffineTranspose operators for pytorch layers
import torch.nn as nn

# Convert flattened input to convolutional input
def convert2to4(x, C): 
    if x.dim() == 1: 
        x = x.unsqueeze(0)
    m,n = x.size()
    in_channels = C.in_channels
    k = int((n//in_channels)**0.5)
    return x.contiguous().view(m,in_channels,k,k)

# Convert convolutional input to flattened input
def convert4to2(x): 
    m = x.size(0)
    return x.contiguous().view(m,-1)

class Affine(nn.Module): 
    def __init__(self): 
        super(Affine, self).__init__()
        self.in_features = None
        self.out_features = None

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
        self.l = nn.Linear(l.in_features, l.out_features, bias=False)
        self.l.weight = l.weight

    def forward(self, x): 
        if x.dim() == 4: 
            x = convert4to2(x)
        out = self.l(x)
        self.record_size(x,out)
        return out

class AffineConv2d(Affine): 
    def __init__(self, l): 
        super(AffineConv2d,self).__init__()
        if not isinstance(l, nn.Conv2d): 
            raise ValueError("Expected nn.Conv2d input.")
        self.l = nn.Conv2d(l.in_channels, 
                           l.out_channels, 
                           l.kernel_size, 
                           stride=l.stride, 
                           padding=l.padding,
                           bias=False)
        self.l.weight = l.weight

    def forward(self, x): 
        if x.dim() == 2: 
            x = convert2to4(x, self.l)
        out = convert4to2(self.l(x))
        self.record_size(x,out)
        return out

class AffineTransposeLinear(nn.Module): 
    def __init__(self, l): 
        super(AffineTransposeLinear, self).__init__()
        if not isinstance(l, nn.Linear):
            raise ValueError("Expected nn.Linear input.")
        self.l = l

    def forward(self, x): 
        if x.dim() == 4: 
            x = convert4to2(x)
        return x.mm(self.l.weight)

class AffineTransposeConv2d(nn.Module): 
    def __init__(self, l): 
        super(AffineTransposeConv2d, self).__init__()
        if not isinstance(l, nn.Conv2d): 
            raise ValueError("Expected nn.Conv2d input.")

        self.l = nn.ConvTranspose2d(l.out_channels, 
                                    l.in_channels, 
                                    l.kernel_size, 
                                    stride=l.stride, 
                                    padding=l.padding,
                                    bias=False)
        self.l.weight = l.weight

    def forward(self, x): 
        if x.dim() == 2: 
            x = convert2to4(x, self.l)
        return convert4to2(self.l(x))