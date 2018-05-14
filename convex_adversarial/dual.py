import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
# import cvxpy as cp

from . import affine as Aff
from . import l1 as L1_engine
from .dense import Dense

import warnings

def batch(A, n): 
    return A.view(n, -1, *A.size()[1:])
def unbatch(A): 
    return A.view(-1, *A.size()[2:])

def select_L(X, k, m, l1_eps, W, l1_type='exact', threshold=None,
             **kwargs):
    if l1_type == 'exact' or k*m > threshold: 
        # print("exact at threshold {}".format(threshold))
        return L1_engine.L1(X, W, **kwargs)
    else: 
        # print("approximate at threshold {}".format(threshold))
        if not isinstance(k, int): 
            raise ValueError('l1 must be an integer')

        if l1_type == 'median': 
            return L1_engine.L1_median(X, k, m, l1_eps, W, **kwargs)

        elif l1_type == 'geometric': 
            return L1_engine.L1_geometric(X, k, m, l1_eps, W, **kwargs)
        else:
            raise ValueError("Unknown l1_type: {}".format(l1_type))

class ForwardPass: 
    def __init__(self, X=None): 
        if X is not None: 
            self.inputs = [X]
        else:
            self.inputs = []
    def apply(self, W): 
        return W(*self.inputs)
    def add(self, X): 
        self.inputs.append(X)
    def add_and_apply(self, X, W): 
        self.add(X)
        return self.apply(W)



class InfBall():
    def __init__(self, X, epsilon): 
        self.epsilon = epsilon

        n = X[0].numel()
        self.nu_x = [X] 
        self.nu_1 = [X.data.new(n,n)]
        torch.eye(n, out=self.nu_1[0])
        self.nu_1[0] = Variable(self.nu_1[0].view(-1,*X.size()[1:]).unsqueeze(0))

    def apply(self, dual_layer): 
        self.nu_x.append(dual_layer.affine(*self.nu_x))
        self.nu_1.append(dual_layer.affine(*self.nu_1))

    def fval(self, nu=None, nu_prev=None): 
        epsilon = self.epsilon
        if nu is None: 
            l1 = self.nu_1[-1].abs().sum(1)
            # nu_x_pos = (self.nu_x[0]-self.epsilon).view(self.nu_x[0].size(0),-1).matmul(self.nu_1[-1].view(784,-1).clamp(min=0))
            # print(nu_x_pos.view(-1))
            # nu_x_pos = (self.nu_x[0]+self.epsilon).view(self.nu_x[0].size(0),-1).matmul(self.nu_1[-1].view(784,-1).clamp(max=0))
            # # print(nu_x_pos.view(-1))
            if isinstance(self.epsilon, Variable): 
                while epsilon.dim() < self.nu_x[-1].dim(): 
                    epsilon = epsilon.unsqueeze(1)

            return (self.nu_x[-1] - epsilon*l1, 
                    self.nu_x[-1] + epsilon*l1)
        else: 
            nu = nu.view(nu.size(0), nu.size(1), -1)
            nu_x = nu.matmul(self.nu_x[0].view(self.nu_x[0].size(0),-1).unsqueeze(2)).squeeze(2)
            if isinstance(self.epsilon, Variable): 
                while epsilon.dim() < nu.dim()-1: 
                    epsilon = epsilon.unsqueeze(1)
                # l1 = self.epsilon
            l1 = epsilon*nu.abs().sum(2)
            return -nu_x - l1

class InfBallBounded():
    def __init__(self, X, epsilon, l=0, u=1): 
        self.epsilon = epsilon
        self.l = (X-epsilon).clamp(min=l).view(X.size(0), 1, -1)
        self.u = (X+epsilon).clamp(max=u).view(X.size(0), 1, -1)

        n = X[0].numel()
        self.nu_x = [X] 
        self.nu_1 = [X.data.new(n,n)]
        torch.eye(n, out=self.nu_1[0])
        self.nu_1[0] = Variable(self.nu_1[0].view(-1,*X.size()[1:]).unsqueeze(0))

    def apply(self, dual_layer): 
        self.nu_x.append(dual_layer.affine(*self.nu_x))
        self.nu_1.append(dual_layer.affine(*self.nu_1))

    def fval(self, nu=None, nu_prev=None): 
        if nu is None: 
            nu = self.nu_1[-1]
            nu_pos = nu.clamp(min=0).view(nu.size(0), nu.size(1), -1)
            nu_neg = nu.clamp(max=0).view(nu.size(0), nu.size(1), -1)

            zu = (self.u.matmul(nu_pos) + self.l.matmul(nu_neg)).squeeze(1)
            zl = (self.u.matmul(nu_neg) + self.l.matmul(nu_pos)).squeeze(1)
            return (zl.view(zl.size(0), *nu.size()[2:]), 
                    zu.view(zu.size(0), *nu.size()[2:]))
        else: 
            nu_pos = nu.clamp(min=0).view(nu.size(0), nu.size(1), -1)
            nu_neg = nu.clamp(max=0).view(nu.size(0), nu.size(1), -1)
            u, l = self.u.unsqueeze(3).squeeze(1), self.l.unsqueeze(3).squeeze(1)
            return (-nu_neg.matmul(l) - nu_pos.matmul(u)).squeeze(2)

class InfBallProj():
    def __init__(self, X, epsilon, k): 
        self.epsilon = epsilon

        n = X[0].numel()
        self.nu_x = [X] 
        self.nu = [Variable(X.data.new(1,k,*X.size()[1:]).cauchy_())]
        # slef.nu_one = [Variable(X.data.new(1,*X.size()[1:]).fill_(1))]

    def apply(self, dual_layer): 
        self.nu_x.append(dual_layer.affine(*self.nu_x))
        self.nu.append(dual_layer.affine(*self.nu))
        # self.nu_one.append(dual_layer.affine(*self.nu_one))

    def fval(self, nu=None, nu_prev=None): 
        if nu is None: 
            l1 = torch.median(self.nu[-1].abs(), 1)[0]
            return (self.nu_x[-1] - self.epsilon*l1, 
                    self.nu_x[-1] + self.epsilon*l1)
        else: 
            nu = nu.view(nu.size(0), nu.size(1), -1)
            nu_x = nu.matmul(self.nu_x[0].view(self.nu_x[0].size(0),-1).unsqueeze(2)).squeeze(2)
            l1 = nu.abs().sum(2)

            return -nu_x - self.epsilon*l1

class InfBallProjBounded():
    def __init__(self, X, epsilon, k, l=0, u=1): 
        self.epsilon = epsilon

        self.nu_one_l = [(X-epsilon).clamp(min=l)]
        self.nu_one_u = [(X+epsilon).clamp(max=u)]
        self.nu_x = [X] 

        self.l = self.nu_one_l[-1].view(X.size(0), 1, -1)
        self.u = self.nu_one_u[-1].view(X.size(0), 1, -1)

        n = X[0].numel()
        R = X.data.new(1,k,*X.size()[1:]).cauchy_()
        self.nu_l = [Variable(R * self.nu_one_l[-1].data.unsqueeze(1))]
        self.nu_u = [Variable(R * self.nu_one_u[-1].data.unsqueeze(1))]

    def apply(self, dual_layer): 
        self.nu_l.append(dual_layer.affine(*self.nu_l))
        self.nu_one_l.append(dual_layer.affine(*self.nu_one_l))
        self.nu_u.append(dual_layer.affine(*self.nu_u))
        self.nu_one_u.append(dual_layer.affine(*self.nu_one_u))

    def fval(self, nu=None, nu_prev=None): 
        if nu is None: 
            nu_l1_u = torch.median(self.nu_u[-1].abs(),1)[0]
            nu_pos_u = (nu_l1_u + self.nu_one_u[-1])/2
            nu_neg_u = (-nu_l1_u + self.nu_one_u[-1])/2

            nu_l1_l = torch.median(self.nu_l[-1].abs(),1)[0]
            nu_pos_l = (nu_l1_l + self.nu_one_l[-1])/2
            nu_neg_l = (-nu_l1_l + self.nu_one_l[-1])/2

            zu = nu_pos_u + nu_neg_l
            zl = nu_neg_u + nu_pos_l
            return zl,zu
        else: 
            return InfBall.fval(self, nu=nu, nu_prev=nu_prev)

class DualLinear(): 
    def __init__(self, layer, out_features): 
        if not isinstance(layer, nn.Linear):
            raise ValueError("Expected nn.Linear input.")
        self.layer = layer
        if layer.bias is None: 
            self.bias = None
        else: 
            self.bias = [Aff.full_bias(layer, out_features[1:])]

    def apply(self, dual_layer): 
        if self.bias is not None: 
            self.bias.append(dual_layer.affine(*self.bias))

    def fval(self, nu=None, nu_prev=None): 
        if nu is None: 
            if self.bias is None: 
                return 0,0
            else: 
                return self.bias[-1], self.bias[-1]
        else:
            if self.bias is None: 
                return 0
            else:
                nu = nu.view(nu.size(0), nu.size(1), -1)
                return -nu.matmul(self.bias[0].view(-1))

    def affine(self, *xs): 
        x = xs[-1]
        return F.linear(x, self.layer.weight)

    def affine_transpose(self, *xs): 
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
        if not isinstance(layer, nn.Conv2d):
            raise ValueError("Expected nn.Conv2d input.")
        self.layer = layer
        if layer.bias is None: 
            self.bias = None
        else: 
            self.bias = [Aff.full_bias(layer, out_features[1:]).contiguous()]

    def affine(self, *xs): 
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

    def affine_transpose(self, *xs): 
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

class DualReshape(): 
    def __init__(self, in_f, out_f): 
        self.in_f = in_f[1:]
        self.out_f = out_f[1:]

    def affine(self, *xs): 
        x = xs[-1]
        shape = x.size()[:-len(self.in_f)] + self.out_f
        return x.view(shape)

    def affine_transpose(self, *xs): 
        x = xs[-1]
        shape = x.size()[:-len(self.out_f)] + self.in_f
        return x.view(shape)

    def apply(self, dual_layer): 
        pass

    def fval(self, nu=None, nu_prev=None): 
        if nu is None: 
            return 0,0
        else:
            return 0

class DualReLU(): 
    def __init__(self, I, d, zl): 
        n = d.data[0].numel()
        if I.data.sum() > 0: 
            self.I_empty = False
            self.I_ind = Variable(I.data.view(-1,n).nonzero())


            self.nus = [Variable(zl.data.new(I.data.sum(), n).zero_())]
            self.nus[-1].scatter_(1, self.I_ind[:,1,None], d[I][:,None])
            self.nus[-1] = self.nus[-1].view(-1, *(d.size()[1:]))
            self.I_collapse = Variable(zl.data.new(self.I_ind.size(0),zl.size(0)).zero_())
            self.I_collapse.scatter_(1, self.I_ind[:,0][:,None], 1)
        else: 
            self.I_empty = True

        self.d = d
        self.I = I
        self.zl = zl

    def apply(self, dual_layer): 
        if self.I_empty: 
            return
        if isinstance(dual_layer, DualReLU): 
            self.nus.append(dual_layer.affine(*self.nus, I_ind=self.I_ind))
        else: 
            self.nus.append(dual_layer.affine(*self.nus))

    def fval(self, nu=None, nu_prev=None): 
        if nu_prev is None:
            if self.I_empty: 
                return 0,0
            nu = self.nus[-1]
            nu = nu.view(nu.size(0), -1)
            zlI = self.zl[self.I]
            zl = (zlI * (-nu.t()).clamp(min=0)).mm(self.I_collapse).t().contiguous()
            zu = -(zlI * nu.t().clamp(min=0)).mm(self.I_collapse).t().contiguous()
            
            zl = zl.view(-1, *(self.nus[-1].size()[1:]))
            zu = zu.view(-1, *(self.nus[-1].size()[1:]))
            return zl,zu
        else: 
            if self.I_empty: 
                return 0
            n = nu_prev.size(0)
            nu = nu_prev.view(n, nu_prev.size(1), -1)
            zl = self.zl.view(n, -1)
            I = self.I.view(n, -1)
            return (nu.clamp(min=0)*zl.unsqueeze(1)).matmul(I.type_as(nu).unsqueeze(2)).squeeze(2)

    def affine(self, *xs, I_ind=None): 
        x = xs[-1]

        d = self.d 
        if x.dim() > d.dim():
            d = d.unsqueeze(1)

        if I_ind is not None: 
            return d[I_ind[:,0]]*x
        else:
            return d*x

    def affine_transpose(self, *xs): 
        return self.affine(*xs)


class DualReLUProj(DualReLU): 
    def __init__(self, I, d, zl, k): 
        n = I.size(0)

        self.d = d
        self.I = I
        self.zl = zl

        if I.data.sum() == 0: 
            warnings.warn('ReLU projection has no origin crossing activations')
            self.I_empty = True
            return
        else:
            self.I_empty = False

        nu = Variable(zl.data.new(n, k, *(d.size()[1:])).zero_())
        nu_one = Variable(zl.data.new(n, *(d.size()[1:])).zero_())
        if  (I.data).sum() > 0: 
            nu.data[I.data.unsqueeze(1).expand_as(nu)] = nu.data.new(I.data.sum()*k).cauchy_()
            nu_one.data[I.data] = 1
        nu = zl.unsqueeze(1)*nu
        nu_one = zl*nu_one

        self.nus = [d.unsqueeze(1)*nu]
        self.nu_ones = [d*nu_one]

    def apply(self, dual_layer): 
        if self.I_empty: 
            return
        self.nus.append(dual_layer.affine(*self.nus))
        self.nu_ones.append(dual_layer.affine(*self.nu_ones))

    def fval(self, nu=None, nu_prev=None): 
        if nu_prev is None:
            if self.I_empty: 
                return 0,0

            n = torch.median(self.nus[-1].abs(), 1)[0]
            no = self.nu_ones[-1]

            # From notes: 
            # \sum_i l_i[nu_i]_+ \approx (-n + no)/2
            # which is the negative of the term for the upper bound
            # for the lower bound, use -nu and negate the output, so 
            # (n - no)/2 since the no term flips twice and the l1 term
            # flips only once. 
            zl = (-n - no)/2
            zu = (n - no)/2

            return zl,zu
        else: 
            return DualReLU.fval(self, nu=nu, nu_prev=nu_prev)

class DualDense(): 
    def __init__(self, dense, dense_t, net, out_features): 
        self.duals = []
        for i,W in enumerate(dense.Ws): 
            if isinstance(W, nn.Conv2d):
                dual_layer = DualConv2d(W, out_features)
            elif isinstance(W, nn.Linear): 
                dual_layer = DualLinear(W, out_features)
            elif isinstance(W, nn.Sequential) and len(W) == 0: 
                dual_layer = DualSequential()
            elif W is None:
                dual_layer = None
            else:
                print(W)
                raise ValueError("Don't know how to parse dense structure")
            self.duals.append(dual_layer)

            if i < len(dense.Ws)-1 and W is not None: 
                idx = i-len(dense.Ws)+1
                # dual_ts needs to be len(dense.Ws)-i long
                net[idx].dual_ts = [dual_layer] + [None]*(len(dense.Ws)-i-len(net[idx].dual_ts)-1) + net[idx].dual_ts

        self.dual_ts = [self.duals[-1]]


    def affine(self, *xs): 
        duals = self.duals[-min(len(xs),len(self.duals)):]
        return sum(W.affine(*xs[:i+1]) 
            for i,W in zip(range(-len(duals) + len(xs), len(xs)),
                duals) if W is not None)

    def affine_transpose(self, *xs): 
        dual_ts = self.dual_ts[-min(len(xs),len(self.dual_ts)):]
        return sum(W.affine_transpose(*xs[:i+1]) 
            for i,W in zip(range(-len(dual_ts) + len(xs), len(xs)),
                dual_ts) if W is not None)

    def apply(self, dual_layer): 
        for W in self.duals: 
            if W is not None: 
                W.apply(dual_layer)

    def fval(self, nu=None, nu_prev=None): 
        fvals = list(W.fval(nu=nu, nu_prev=nu_prev) for W in self.duals if W is
            not None)
        if nu is None: 
            l,u = zip(*fvals)
            return sum(l), sum(u)
        else:
            return sum(fvals)


class DualBatchNorm2d(): 
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

        self.D = (weight/Variable(denom)).unsqueeze(1).unsqueeze(2)
        self.ds = [((bias - weight*Variable(mu/denom)).unsqueeze(1).unsqueeze
            (2)).expand(out_features[1:]).contiguous()]
        

    def affine(self, *xs): 
        x = xs[-1]
        return self.D*x

    def affine_transpose(self, *xs): 
        return self.affine(*xs)

    def apply(self, dual_layer): 
        self.ds.append(dual_layer.affine(*self.ds))

    def fval(self, nu=None, nu_prev=None): 

        if nu is None: 
            d = self.ds[-1]
            return d, d
        else:
            d = self.ds[0].view(-1)
            nu = nu.view(nu.size(0), nu.size(1), -1)
            return -nu.matmul(d)

class DualSequential(): 
    def affine(self, *xs): 
        return xs[-1]

    def affine_transpose(self, *xs): 
        return xs[-1]

    def apply(self, dual_layer): 
        pass

    def fval(self, nu=None, nu_prev=None): 
        if nu is None: 
            return 0,0
        else:
            return 0

class DualNetBounds: 
    def __init__(self, net, X, epsilon, alpha_grad=False, scatter_grad=False, 
                 l1_proj=None, l1_eps=None, m=None,
                 l1_type='exact', bounded_input=False):
        """ 
        net : ReLU network
        X : minibatch of examples
        epsilon : size of l1 norm ball to be robust against adversarial examples
        alpha_grad : flag to propagate gradient through alpha
        scatter_grad : flag to propagate gradient through scatter operation
        l1 : size of l1 projection
        l1_eps : the bound is correct up to a 1/(1-l1_eps) factor
        m : number of probabilistic bounds to take the max over
        """
        # need to change that if no batchnorm, can pass just a single example
        if any('BatchNorm2d' in str(l.__class__.__name__) for l in net): 
            zs = [Variable(X.data, volatile=True)]
        else:
            zs = [Variable(X.data[:1], volatile=True)]
        nf = [zs[0].size()]
        for l in net: 
            if isinstance(l, Dense): 
                zs.append(l(*zs))
            else:
                zs.append(l(zs[-1]))
            nf.append(zs[-1].size())


        # Use the bounded boxes
        if l1_proj is not None and l1_type=='median' and X[0].numel() > l1_proj:
            if bounded_input: 
                dual_net = [InfBallProjBounded(X,epsilon,l1_proj)]
            else: 
                dual_net = [InfBallProj(X,epsilon,l1_proj)]
        else:
            if bounded_input: 
                dual_net = [InfBallBounded(X, epsilon)]
            else:
                dual_net = [InfBall(X, epsilon)]

        if any(isinstance(l, Dense) for l in net): 
            dense_t = Aff.transpose_all(net)

        for i,(in_f,out_f,layer) in enumerate(zip(nf[:-1], nf[1:], net)): 
            if isinstance(layer, nn.Linear): 
                dual_layer = DualLinear(layer, out_f)
            elif isinstance(layer, nn.Conv2d): 
                dual_layer = DualConv2d(layer, out_f)
            elif isinstance(layer, nn.ReLU): 
                zl, zu = zip(*[l.fval() for l in dual_net])
                zl, zu = sum(zl), sum(zu)

                d = (zl >= 0).detach().type_as(X)
                I = ((zu > 0).detach() * (zl < 0).detach())
                if I.data.sum() > 0:
                    d[I] += zu[I]/(zu[I] - zl[I])

                if l1_proj is not None and l1_type=='median' and I.data.sum() > l1_proj:
                    dual_layer = DualReLUProj(I, d, zl, l1_proj)
                else:
                    dual_layer = DualReLU(I, d, zl)

            elif 'Flatten' in (str(layer.__class__.__name__)): 
                dual_layer = DualReshape(in_f, out_f)
            elif isinstance(layer, Dense): 
                assert isinstance(dense_t[i], Dense)
                dual_layer = DualDense(layer, dense_t[i], dual_net, out_f)
            elif isinstance(layer, nn.BatchNorm2d):
                dual_layer = DualBatchNorm2d(layer, zs[i], out_f)
            else:
                print(layer)
                raise ValueError("No module for layer {}".format(str(layer.__class__.__name__)))

            # skip last layer
            if i < len(net)-1: 
                for l in dual_net: 
                    l.apply(dual_layer)
                dual_net.append(dual_layer)
            else: 
                self.last_layer = dual_layer

        self.dual_net = dual_net
        return 
        
    def g(self, c):
        nu = [-c]
        nu.append(self.last_layer.affine_transpose(nu[0]))
        for l in reversed(self.dual_net[1:]): 
            nu.append(l.affine_transpose(*nu))
        dual_net = self.dual_net + [self.last_layer]
        
        nu.append(None)
        nu = list(reversed(nu))

        return sum(l.fval(nu=n, nu_prev=nprev) 
            for l,nprev,n in zip(dual_net, nu[:-1],nu[1:]))

class RobustBounds(nn.Module): 
    def __init__(self, net, epsilon, **kwargs): 
        super(RobustBounds, self).__init__()
        self.net = net
        self.epsilon = epsilon
        self.kwargs = kwargs

    def forward(self, X,y): 
        num_classes = self.net[-1].out_features
        dual = DualNetBounds(self.net, X, self.epsilon, **self.kwargs)
        c = Variable(torch.eye(num_classes).type_as(X.data)[y.data].unsqueeze(1) - torch.eye(num_classes).type_as(X.data).unsqueeze(0))
        if X.is_cuda:
            c = c.cuda()
        f = -dual.g(c)
        return f

def robust_loss(net, epsilon, X, y, 
                size_average=True, device_ids=None, **kwargs):
    f = nn.DataParallel(RobustBounds(net, epsilon, **kwargs),
                        device_ids=None)(X,y)
    err = (f.data.max(1)[1] != y.data)
    if size_average: 
        err = err.sum()/X.size(0)
    ce_loss = nn.CrossEntropyLoss(size_average=size_average)(f, y)
    return ce_loss, err