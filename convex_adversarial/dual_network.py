import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from .utils import Dense, DenseSequential
from .dual_inputs import select_input
from .dual_layers import select_layer

import warnings


class DualNetwork(nn.Module):   
    def __init__(self, net, X, epsilon, 
                 proj=None, norm_type='l1', bounded_input=False, 
                 data_parallel=True):
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
        dual_net = [select_input(X, epsilon, proj, norm_type, bounded_input)]

        for i,(in_f,out_f,layer) in enumerate(zip(nf[:-1], nf[1:], net)): 
            dual_layer = select_layer(layer, dual_net, X, proj, norm_type,
                                      in_f, out_f, zs[i])

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
                size_average=True, device_ids=None, parallel=False, **kwargs):
    reduction = 'mean' if size_average else 'none'
    if parallel: 
        f = nn.DataParallel(RobustBounds(net, epsilon, **kwargs))(X,y)
    else: 
        f = RobustBounds(net, epsilon, **kwargs)(X,y)
    err = (f.max(1)[1] != y)
    if size_average: 
        err = err.sum().item()/X.size(0)
    ce_loss = nn.CrossEntropyLoss(reduction=reduction)(f, y)
    return ce_loss, err

class InputSequential(nn.Sequential): 
    def __init__(self, *args, **kwargs): 
        self.i = 0
        super(InputSequential, self).__init__(*args, **kwargs)

    def set_start(self, i): 
        self.i = i

    def forward(self, input): 
        """ Helper class to apply a sequential model starting at the ith layer """
        xs = [input]
        for j,module in enumerate(self._modules.values()): 
            if j >= self.i: 
                if 'Dense' in type(module).__name__:
                    xs.append(module(*xs))
                else:
                    xs.append(module(xs[-1]))
        return xs[-1]


# Data parallel versions of the loss calculation
def robust_loss_parallel(net, epsilon, X, y, proj=None, 
                 norm_type='l1', bounded_input=False, size_average=True): 
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

    dual_net = [select_input(X, epsilon, proj, norm_type, bounded_input)]

    for i,(in_f,out_f,layer) in enumerate(zip(nf[:-1], nf[1:], net)): 
        if isinstance(layer, nn.ReLU): 
            # compute bounds
            D = (InputSequential(*dual_net[1:]))
            Dp = nn.DataParallel(D)
            zl,zu = 0,0
            for j,dual_layer in enumerate(dual_net): 
                D.set_start(j)
                out = dual_layer.bounds(network=Dp)
                zl += out[0]
                zu += out[1]

            dual_layer = select_layer(layer, dual_net, X, proj, norm_type,
                in_f, out_f, zs[i], zl=zl, zu=zu)
        else:
            dual_layer = select_layer(layer, dual_net, X, proj, norm_type,
                in_f, out_f, zs[i])
        
        dual_net.append(dual_layer)

    num_classes = net[-1].out_features
    c = Variable(torch.eye(num_classes).type_as(X)[y].unsqueeze(1) - torch.eye(num_classes).type_as(X).unsqueeze(0))
    if X.is_cuda:
        c = c.cuda()

    # same as f = -dual.g(c)
    nu = [-c]
    for l in reversed(dual_net[1:]): 
        nu.append(l.T(*nu))
    
    f = -sum(l.objective(*nu[:min(len(dual_net)-i+1, len(dual_net))]) 
             for i,l in enumerate(dual_net))

    err = (f.max(1)[1] != y)

    if size_average: 
        err = err.sum().item()/X.size(0)
    ce_loss = nn.CrossEntropyLoss(reduce=size_average)(f, y)
    return ce_loss, err