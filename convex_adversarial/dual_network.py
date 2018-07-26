import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from .utils import Dense, DenseSequential, transpose_all
from .dual_inputs import select_input
from .dual_layers import select_layer

import warnings


class DualNetwork(nn.Module):   
    def __init__(self, net, X, epsilon, 
                 l1_proj=None, l1_type='exact', bounded_input=False):
        """ 
        This class creates the dual network. 

        net : ReLU network
        X : minibatch of examples
        epsilon : size of l1 norm ball to be robust against adversarial examples
        alpha_grad : flag to propagate gradient through alpha
        scatter_grad : flag to propagate gradient through scatter operation
        l1 : size of l1 projection
        l1_eps : the bound is correct up to a 1/(1-l1_eps) factor
        m : number of probabilistic bounds to take the max over
        """
        super(DualNetwork, self).__init__()
        # need to change that if no batchnorm, can pass just a single example
        if not isinstance(net, (nn.Sequential, DenseSequential)): 
            raise ValueError("Network must be a nn.Sequential or DenseSequential module")
        with torch.no_grad(): 
            if any('BatchNorm2d' in str(l.__class__.__name__) for l in net): 
                zs = [X]
            else:
                zs = [X[:1]]
            nf = [zs[0].size()]
            for l in net: 
                if isinstance(l, Dense): 
                    zs.append(l(*zs))
                else:
                    zs.append(l(zs[-1]))
                    nf.append(zs[-1].size())


        # Use the bounded boxes
        dual_net = [select_input(X, epsilon, l1_proj, l1_type, bounded_input)]

        if any(isinstance(l, Dense) for l in net): 
            dense_t = transpose_all(net)
        else: 
            dense_t = [None]*len(net)

        for i,(in_f,out_f,layer) in enumerate(zip(nf[:-1], nf[1:], net)): 
            dual_layer = select_layer(layer, dual_net, X, l1_proj, l1_type, in_f, out_f, dense_t[i], zs[i])

            # skip last layer
            if i < len(net)-1: 
                for l in dual_net: 
                    l.apply(dual_layer)
                dual_net.append(dual_layer)
            else: 
                self.last_layer = dual_layer

        self.dual_net = dual_net
        return 

    def forward(self, c):
        """ For the constructed given dual network, compute the objective for
        some given vector c """
        nu = [-c]
        nu.append(self.last_layer.T(*nu))
        for l in reversed(self.dual_net[1:]): 
            nu.append(l.T(*nu))
        dual_net = self.dual_net + [self.last_layer]
        
        return sum(l.objective(*nu[:min(len(dual_net)-i+1, len(dual_net))]) for
           i,l in enumerate(dual_net))

class DualNetBounds(DualNetwork): 
    def __init__(self, *args, **kwargs):
        warnings.warn("DualNetBounds is deprecated. Use the proper "
                      "PyTorch module DualNetwork instead. ")
        super(DualNetBounds, self).__init__(*args, **kwargs)

    def g(self, c):
        return self(c)

class RobustBounds(nn.Module): 
    def __init__(self, net, epsilon, **kwargs): 
        super(RobustBounds, self).__init__()
        self.net = net
        self.epsilon = epsilon
        self.kwargs = kwargs

    def forward(self, X,y): 
        num_classes = self.net[-1].out_features
        dual = DualNetwork(self.net, X, self.epsilon, **self.kwargs)
        c = Variable(torch.eye(num_classes).type_as(X)[y].unsqueeze(1) - torch.eye(num_classes).type_as(X).unsqueeze(0))
        if X.is_cuda:
            c = c.cuda()
        f = -dual(c)
        return f

def robust_loss(net, epsilon, X, y, 
                size_average=True, device_ids=None, **kwargs):
    f = nn.DataParallel(RobustBounds(net, epsilon, **kwargs))(X,y)
    err = (f.max(1)[1] != y)
    if size_average: 
        err = err.sum().item()/X.size(0)
    ce_loss = nn.CrossEntropyLoss(reduce=size_average)(f, y)
    return ce_loss, err

# Data parallel versions of the loss calculation
# Warning: experimental

class DualSequential(nn.Module): 
    def __init__(self, dual_layers, net): 
        super(DualSequential, self).__init__()
        self.dual_layers = nn.ModuleList(dual_layers)
        self.net = net
        pass
    def forward(self, x, I_ind=None): 
        zs = [x]
        for l in list(self.dual_layers)[1:]: 
            if isinstance(l, DualDense): 
                zs.append(l(*zs))
            elif isinstance(l, DualReLU): 
                zs.append(l(zs[-1], I_ind=I_ind))
            else:
                zs.append(l(zs[-1]))
        return zs[-1]

def dual_helper(dual_layer, D): 
    if isinstance(dual_layer, (DualLinear, DualConv2d)): 
        b = dual_layer.bias[0]
        if isinstance(dual_layer, DualConv2d): 
            b = b.unsqueeze(0)
        Db = D(b)
        return Db, Db
    elif isinstance(dual_layer, DualReshape):
        return 0,0
    elif isinstance(dual_layer, DualReLU):
        if dual_layer.I_empty: 
            return 0,0
        D = nn.DataParallel(D)
        nu0 = D(dual_layer.nus[0], I_ind=dual_layer.I_ind)

        nu = nu0.view(nu0.size(0), -1)
        zlI = dual_layer.zl[dual_layer.I]
        zl = (zlI * (-nu.t()).clamp(min=0)).mm(dual_layer.I_collapse).t().contiguous()
        zu = -(zlI * nu.t().clamp(min=0)).mm(dual_layer.I_collapse).t().contiguous()
        
        return zl.view(-1, *(nu0.size()[1:])), zu.view(-1, *(nu0.size()[1:]))
    elif isinstance(dual_layer, DualDense): 
        fvals = [dual_helper(d, D) for d in dual_layer.duals if d is not None]
        l,u = zip(*fvals)
        return sum(l), sum(u)
    else: 
        print(dual_layer)
        raise NotImplementedError

def robust_loss_parallel(net, epsilon, X, y, l1_proj=None, 
                 l1_type='exact', bounded_input=False, size_average=True): 
    if any('BatchNorm2d' in str(l.__class__.__name__) for l in net): 
        raise NotImplementedError
    if bounded_input: 
        raise NotImplementedError('parallel loss for bounded input spaces not implemented')
    if X.size(0) != 1: 
        raise ValueError('Only use this function for a single example. This is '
            'intended for the use case when a single example does not fit in '
            'memory.')
    zs = [X[:1]]
    nf = [zs[0].size()]
    for l in net: 
        if isinstance(l, Dense): 
            zs.append(l(*zs))
        else:
            zs.append(l(zs[-1]))
        nf.append(zs[-1].size())

    dual_net = [select_input(X, epsilon, l1_proj, l1_type, bounded_input)]

    if any(isinstance(l, Dense) for l in net): 
        dense_t = Aff.transpose_all(net)
    else: 
        dense_t = [None]*len(net)

    eye = dual_net[0].nu_1[0]
    x = dual_net[0].nu_x[0]

    for i,(in_f,out_f,layer) in enumerate(zip(nf[:-1], nf[1:], net)): 
        if isinstance(layer, nn.ReLU): 
            # compute bounds
            D =nn.DataParallel(DualSequential(dual_net, net))

            # should be negative, but doesn't matter with abs()
            nu_1 = D(dual_net[0].nu_1[0]).abs().sum(1)
            nu_x = D(dual_net[0].nu_x[0])

            rest = 0
            rest_l = 0
            rest_u = 0
            for i,dual_layer in enumerate(dual_net[1:]): 
                D = DualSequential(dual_net[i+1:], net)
                out = dual_helper(dual_layer, D)
                rest_l += out[0]
                rest_u += out[1]

            zl = nu_x - epsilon*nu_1 + rest_l
            zu = nu_x + epsilon*nu_1 + rest_u

            d = (zl >= 0).detach().type_as(X)
            I = ((zu > 0).detach() * (zl < 0).detach())
            if I.sum().item() > 0:
                d[I] += zu[I]/(zu[I] - zl[I])

            dual_layer = DualReLU(I, d, zl)
        else:
            dual_layer = select_layer(layer, dual_net, X, l1_proj, l1_type, in_f, out_f, dense_t[i], zs[i])
        
        dual_net.append(dual_layer)


    num_classes = net[-1].out_features
    c = Variable(torch.eye(num_classes).type_as(X)[y].unsqueeze(1) - torch.eye(num_classes).type_as(X).unsqueeze(0))
    if X.is_cuda:
        c = c.cuda()

    # same as f = -dual.g(c)
    nu = [-c]
    for l in reversed(dual_net[1:]): 
        nu.append(l.T(*nu))
    
    nu.append(None)
    nu = list(reversed(nu))
    f = -sum(l.fval(nu=n, nu_prev=nprev) 
        for l,nprev,n in zip(dual_net, nu[:-1],nu[1:]))

    err = (f.max(1)[1] != y)

    if size_average: 
        err = err.sum().item()/X.size(0)
    ce_loss = nn.CrossEntropyLoss(reduce=size_average)(f, y)
    return ce_loss, err