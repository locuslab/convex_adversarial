import torch
import torch.nn as nn
import torch.nn.functional as F

from .dual import DualLayer
from .utils import full_bias, Dense

def select_layer(layer, dual_net, X, l1_proj, l1_type, in_f, out_f, zsi,
                 zl=None, zu=None):
    if isinstance(layer, nn.Linear): 
        return DualLinear(layer, out_f)
    elif isinstance(layer, nn.Conv2d): 
        return DualConv2d(layer, out_f)
    elif isinstance(layer, nn.ReLU):   
        if zl is None and zu is None:
            zl, zu = zip(*[l.bounds() for l in dual_net])
            # for l,dn in zip(zl,dual_net):
            #     print(dn, l.size())
            zl, zu = sum(zl), sum(zu)
        if zl is None or zu is None: 
            raise ValueError("Must either provide both l,u bounds or neither.")

        I = ((zu > 0).detach() * (zl < 0).detach())
        if l1_proj is not None and l1_type=='median' and I.sum().item() > l1_proj:
            return DualReLUProj(zl, zu, l1_proj)
        else:
            return DualReLU(zl, zu)

    elif 'Flatten' in (str(layer.__class__.__name__)): 
        return DualReshape(in_f, out_f)
    elif isinstance(layer, Dense): 
        return DualDense(layer, dual_net, out_f)
    elif isinstance(layer, nn.BatchNorm2d):
        return DualBatchNorm2d(layer, zsi, out_f)
    else:
        print(layer)
        raise ValueError("No module for layer {}".format(str(layer.__class__.__name__)))

def batch(A, n): 
    return A.view(n, -1, *A.size()[1:])
def unbatch(A): 
    return A.view(-1, *A.size()[2:])

class DualLinear(DualLayer): 
    def __init__(self, layer, out_features): 
        super(DualLinear, self).__init__()
        if not isinstance(layer, nn.Linear):
            raise ValueError("Expected nn.Linear input.")
        self.layer = layer
        if layer.bias is None: 
            self.bias = None
        else: 
            self.bias = [full_bias(layer, out_features[1:])]

    def apply(self, dual_layer): 
        if self.bias is not None: 
            self.bias.append(dual_layer(*self.bias))

    def bounds(self, network=None):
        if self.bias is None: 
            return 0,0
        else: 
            if network is None: 
                b = self.bias[-1]
            else:
                b = network(self.bias[0])
            if b is None:
                return 0,0
            return b,b

    def objective(self, *nus): 
        if self.bias is None: 
            return 0
        else:
            nu = nus[-2]
            nu = nu.view(nu.size(0), nu.size(1), -1)
            return -nu.matmul(self.bias[0].view(-1))

    def forward(self, *xs): 
        x = xs[-1]
        if x is None: 
            return None
        return F.linear(x, self.layer.weight)

    def T(self, *xs): 
        x = xs[-1]
        return F.linear(x, self.layer.weight.t())

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

class DualConv2d(DualLinear): 
    def __init__(self, layer, out_features): 
        super(DualLinear, self).__init__()
        if not isinstance(layer, nn.Conv2d):
            raise ValueError("Expected nn.Conv2d input.")
        self.layer = layer
        if layer.bias is None: 
            self.bias = None
        else: 
            self.bias = [full_bias(layer, out_features[1:]).contiguous()]

    def forward(self, *xs): 
        x = xs[-1]
        if xs[-1].dim() == 5:  
            n = x.size(0)
            x = unbatch(x)
        out = conv2d(x, self.layer.weight, 
                       stride=self.layer.stride,
                       padding=self.layer.padding)
        if xs[-1].dim() == 5:  
            out = batch(out, n)
        return out

    def T(self, *xs): 
        x = xs[-1]
        if xs[-1].dim() == 5:  
            n = x.size(0)
            x = unbatch(x)
        out = conv_transpose2d(x, self.layer.weight, 
                                 stride=self.layer.stride,
                                 padding=self.layer.padding)
        if xs[-1].dim() == 5:  
            out = batch(out, n)
        return out

class DualReshape(DualLayer): 
    def __init__(self, in_f, out_f): 
        super(DualReshape, self).__init__()
        self.in_f = in_f[1:]
        self.out_f = out_f[1:]

    def forward(self, *xs): 
        x = xs[-1]
        shape = x.size()[:-len(self.in_f)] + self.out_f
        return x.view(shape)

    def T(self, *xs): 
        x = xs[-1]
        shape = x.size()[:-len(self.out_f)] + self.in_f
        return x.view(shape)

    def apply(self, dual_layer): 
        pass

    def bounds(self, network=None): 
        return 0,0

    def objective(self, *nus): 
        return 0

class DualReLU(DualLayer): 
    def __init__(self, zl, zu): 
        super(DualReLU, self).__init__()


        d = (zl >= 0).detach().type_as(zl)
        I = ((zu > 0).detach() * (zl < 0).detach())
        if I.sum().item() > 0:
            d[I] += zu[I]/(zu[I] - zl[I])

        n = d[0].numel()
        if I.sum().item() > 0: 
            self.I_empty = False
            self.I_ind = I.view(-1,n).nonzero()


            self.nus = [zl.new(I.sum().item(), n).zero_()]
            self.nus[-1].scatter_(1, self.I_ind[:,1,None], d[I][:,None])
            self.nus[-1] = self.nus[-1].view(-1, *(d.size()[1:]))
            self.I_collapse = zl.new(self.I_ind.size(0),zl.size(0)).zero_()
            self.I_collapse.scatter_(1, self.I_ind[:,0][:,None], 1)
        else: 
            self.I_empty = True

        self.d = d
        self.I = I
        self.zl = zl
        self.zu = zu

    def apply(self, dual_layer): 
        if self.I_empty: 
            return
        if isinstance(dual_layer, DualReLU): 
            self.nus.append(dual_layer(*self.nus, I_ind=self.I_ind))
        else: 
            self.nus.append(dual_layer(*self.nus))

    def bounds(self, network=None): 
        if self.I_empty: 
            return 0,0
        if network is None: 
            nu = self.nus[-1]
        else:
            nu = network(self.nus[0])
        if nu is None: 
            return 0,0
        size = nu.size()
        nu = nu.view(nu.size(0), -1)
        zlI = self.zl[self.I]
        zl = (zlI * (-nu.t()).clamp(min=0)).mm(self.I_collapse).t().contiguous()
        zu = -(zlI * nu.t().clamp(min=0)).mm(self.I_collapse).t().contiguous()

        zl = zl.view(-1, *(size[1:]))
        zu = zu.view(-1, *(size[1:]))
        return zl,zu

    def objective(self, *nus): 
        nu_prev = nus[-1]
        if self.I_empty: 
            return 0
        n = nu_prev.size(0)
        nu = nu_prev.view(n, nu_prev.size(1), -1)
        zl = self.zl.view(n, -1)
        I = self.I.view(n, -1)
        return (nu.clamp(min=0)*zl.unsqueeze(1)).matmul(I.type_as(nu).unsqueeze(2)).squeeze(2)


    def forward(self, *xs, I_ind=None): 
        x = xs[-1]
        if x is None:
            return None

        if self.d.is_cuda:
            d = self.d.cuda(device=x.get_device())
        else:
            d = self.d
        if x.dim() > d.dim():
            d = d.unsqueeze(1)

        if I_ind is not None: 
            I_ind = I_ind.to(dtype=torch.long, device=x.device)
            return d[I_ind[:,0]]*x
        else:
            return d*x

    def T(self, *xs): 
        return self(*xs)


class DualReLUProj(DualReLU): 
    def __init__(self, zl, zu, k): 
        DualLayer.__init__(self)
        d = (zl >= 0).detach().type_as(zl)
        I = ((zu > 0).detach() * (zl < 0).detach())
        if I.sum().item() > 0:
            d[I] += zu[I]/(zu[I] - zl[I])

        n = I.size(0)

        self.d = d
        self.I = I
        self.zl = zl
        self.zu = zu

        if I.sum().item() == 0: 
            warnings.warn('ReLU projection has no origin crossing activations')
            self.I_empty = True
            return
        else:
            self.I_empty = False

        nu = zl.new(n, k, *(d.size()[1:])).zero_()
        nu_one = zl.new(n, *(d.size()[1:])).zero_()
        if  I.sum() > 0: 
            nu[I.unsqueeze(1).expand_as(nu)] = nu.new(I.sum().item()*k).cauchy_()
            nu_one[I] = 1
        nu = zl.unsqueeze(1)*nu
        nu_one = zl*nu_one

        self.nus = [d.unsqueeze(1)*nu]
        self.nu_ones = [d*nu_one]

    def apply(self, dual_layer): 
        if self.I_empty: 
            return
        self.nus.append(dual_layer(*self.nus))
        self.nu_ones.append(dual_layer(*self.nu_ones))

    def bounds(self, network=None): 
        if self.I_empty: 
            return 0,0

        if network is None: 
            nu = self.nus[-1]
            no = self.nu_ones[-1]
        else: 
            nu = network(self.nus[0])
            no = network(self.nu_ones[0])

        n = torch.median(self.nus[-1].abs(), 1)[0]

        # From notes: 
        # \sum_i l_i[nu_i]_+ \approx (-n + no)/2
        # which is the negative of the term for the upper bound
        # for the lower bound, use -nu and negate the output, so 
        # (n - no)/2 since the no term flips twice and the l1 term
        # flips only once. 
        zl = (-n - no)/2
        zu = (n - no)/2

        return zl,zu

class DualDense(DualLayer): 
    def __init__(self, dense, net, out_features): 
        super(DualDense, self).__init__()
        self.duals = nn.ModuleList([])
        for i,W in enumerate(dense.Ws): 
            if isinstance(W, nn.Conv2d):
                dual_layer = DualConv2d(W, out_features)
            elif isinstance(W, nn.Linear): 
                dual_layer = DualLinear(W, out_features)
            elif isinstance(W, nn.Sequential) and len(W) == 0: 
                dual_layer = Identity()
            elif W is None:
                dual_layer = None
            else:
                print(W)
                raise ValueError("Don't know how to parse dense structure")
            self.duals.append(dual_layer)

            if i < len(dense.Ws)-1 and W is not None: 
                idx = i-len(dense.Ws)+1
                # dual_ts needs to be len(dense.Ws)-i long
                net[idx].dual_ts = nn.ModuleList([dual_layer] + [None]*(len(dense.Ws)-i-len(net[idx].dual_ts)-1) + list(net[idx].dual_ts))

        self.dual_ts = nn.ModuleList([self.duals[-1]])


    def forward(self, *xs): 
        duals = list(self.duals)[-min(len(xs),len(self.duals)):]
        if all(W is None for W in duals): 
            return None
        out = [W(*xs[:i+1]) 
            for i,W in zip(range(-len(duals) + len(xs), len(xs)),
                duals) if W is not None]
        return sum(o for o in out if o is not None)

    def T(self, *xs): 
        dual_ts = list(self.dual_ts)[-min(len(xs),len(self.dual_ts)):]
        if all(W is None for W in dual_ts): 
            return None
        return sum(W.T(*xs[:i+1]) 
            for i,W in zip(range(-len(dual_ts) + len(xs), len(xs)),
                dual_ts) if W is not None)

    def apply(self, dual_layer): 
        for W in self.duals: 
            if W is not None: 
                W.apply(dual_layer)

    def bounds(self, network=None): 
        fvals = list(W.bounds(network=network) for W in self.duals 
                        if W is not None)
        l,u = zip(*fvals)
        return sum(l), sum(u)

    def objective(self, *nus): 
        fvals = list(W.objective(*nus) for W in self.duals if W is not None)
        return sum(fvals)

class DualBatchNorm2d(DualLayer): 
    def __init__(self, layer, minibatch, out_features): 
        if layer.training: 
            minibatch = minibatch.data.transpose(0,1).contiguous()
            minibatch = minibatch.view(minibatch.size(0), -1)
            mu = minibatch.mean(1)
            var = minibatch.var(1)
        else: 
            mu = layer.running_mean
            var = layer.running_var
        
        eps = layer.eps

        weight = layer.weight
        bias = layer.bias
        denom = torch.sqrt(var + eps)

        self.D = (weight/denom).unsqueeze(1).unsqueeze(2)
        self.ds = [((bias - weight*mu/denom).unsqueeze(1).unsqueeze
            (2)).expand(out_features[1:]).contiguous()]
        

    def forward(self, *xs): 
        x = xs[-1]
        return self.D*x

    def T(self, *xs): 
        return self(*xs)

    def apply(self, dual_layer): 
        self.ds.append(dual_layer(*self.ds))

    def bounds(self, network=None):
        if network is None:
            d = self.ds[-1]
        else:
            d = network(self.ds[0])
        return d, d

    def objective(self, *nus): 
        nu = nus[-2]
        d = self.ds[0].view(-1)
        nu = nu.view(nu.size(0), nu.size(1), -1)
        return -nu.matmul(d)

class Identity(DualLayer): 
    def forward(self, *xs): 
        return xs[-1]

    def T(self, *xs): 
        return xs[-1]

    def apply(self, dual_layer): 
        pass

    def bounds(self, network=None): 
        return 0,0

    def objective(self, *nus): 
        return 0