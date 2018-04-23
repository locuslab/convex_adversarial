import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
# import cvxpy as cp

from . import affine as Aff
from . import l1 as L1_engine

def AffineTranspose(l): 
    if isinstance(l, nn.Linear): 
        return Aff.AffineTransposeLinear(l)
    elif isinstance(l, nn.Conv2d): 
        return Aff.AffineTransposeConv2d(l)
    else:
        raise ValueError("AffineTranspose class not found for given layer.")

def Affine(l): 
    if isinstance(l, nn.Linear): 
        return Aff.AffineLinear(l)
    elif isinstance(l, nn.Conv2d): 
        return Aff.AffineConv2d(l)
    else:
        raise ValueError("Affine class not found for given layer.")

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
    else:
        raise ValueError("Full bias can't be formed for given layer.")

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

class DualNetBounds: 
    def __init__(self, net, X, epsilon, alpha_grad=False, scatter_grad=False, 
                 l1_proj=None, l1_eps=None, m=None, 
                 l1_type='exact'):
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

        n = X.size(0)
        self.layers = [l for l in net if isinstance(l, (nn.Linear, nn.Conv2d))]
        self.affine_transpose = [AffineTranspose(l) for l in self.layers]
        self.affine = [Affine(l) for l in self.layers]

        self.k = len(self.layers)+1

        # initialize affine layers with a forward pass
        _ = X[0].view(1,-1)
        for a in self.affine: 
            _ = a(_)

        # l1_median = None
        # l1_geometric = 100
        # if l1_proj is not None and median: 
        #     if not isinstance(l1_proj, int): 
        #         raise ValueError('l1 must be an integer')
        #     L = L1_engine.L1_median(X, l1_proj, m, l1_eps)
        #     kwargs = { }
        # elif l1_proj is not None and geometric: 
        #     if not isinstance(l1_proj, int): 
        #         raise ValueError('l1 must be an integer')
        #     # should change this to only use projection if # of activations > k
        #     L = L1_engine.L1_geometric(X, l1_proj, m, l1_eps)
        #     kwargs = { } 
        # else: 
        #     L = L1_engine.L1(X)
        #     kwargs = {}

        self.biases = [full_bias(l, self.affine[i].out_features) 
                        for i,l in enumerate(self.layers)]
        
        gamma = [self.biases[0]]
        nu_hat_x = self.affine[0](X)

        L0 = select_L(X, l1_proj, m, l1_eps, self.affine[0], l1_type=l1_type,
                      threshold=self.affine[0].in_features)
        l1 = L0.l1_norm()
        # eye = L.input(self.affine[0].in_features, **kwargs)
        # nu_hat_1 = self.affine[0](eye).unsqueeze(0)
        # l1 = L.l1_norm(nu_hat_1)
        self.zl = [nu_hat_x + gamma[0] - epsilon*l1]
        self.zu = [nu_hat_x + gamma[0] + epsilon*l1]

        self.I = []
        self.I_empty = []
        self.I_neg = []
        self.I_pos = []
        self.X = X
        self.epsilon = epsilon
        I_collapse = []
        Ls = []

        # set flags
        self.scatter_grad = scatter_grad
        self.alpha_grad = alpha_grad

        for i in range(0,self.k-2):
            # compute sets and activation
            self.I_neg.append((self.zu[-1] <= 0).detach())
            self.I_pos.append((self.zl[-1] > 0).detach())
            self.I.append(((self.zu[-1] > 0) * (self.zl[-1] < 0)).detach())
            self.I_empty.append(self.I[-1].data.long().sum() == 0)
            
            I_nonzero = ((self.zu[-1]!=self.zl[-1])*self.I[-1]).detach()
            d = self.I_pos[-1].type_as(X).clone()

            # Avoid division by zero by indexing with I_nonzero
            if I_nonzero.data.sum() > 0:
                d[I_nonzero] += self.zu[-1][I_nonzero]/(self.zu[-1][I_nonzero] - self.zl[-1][I_nonzero])

            for L in Ls: 
                if L is not None: 
                    L.apply(self.affine[i+1], d)
            if not self.I_empty[-1]: 
                Ls.append(select_L(X, l1_proj, m, l1_eps, self.affine[i+1],
                                   l1_type=l1_type,
                                   threshold=self.I[-1].data.sum()/X.size(0),
                                   I=self.I[-1], d=d,
                                   scatter_grad=scatter_grad, zl=self.zl[-1]))
            else: 
                Ls.append(None)

            # initialize new terms
            # L.apply(self.affine[i+1], d)
            # if not self.I_empty[-1]:
            #     L.add_layer(self.affine[i+1], self.I, d, 
            #                         scatter_grad, **kwargs)
            # else:
            #     L.skip()

            gamma.append(self.biases[i+1])
            # propagate bias terms
            for j in range(0,i+1):
                gamma[j] = self.affine[i+1](d * gamma[j])
                
            nu_hat_x = self.affine[i+1](d*nu_hat_x)
            # nu_hat_1 = batch(self.affine[i+1](unbatch(d.unsqueeze(1)*nu_hat_1)), n)
            # l1 = L0.l1_norm(nu_hat_1)
            L0.apply(self.affine[i+1], d)
            l1 = L0.l1_norm()

            # nu_zl, nu_zu = L.nu_zlu(self.zl, *args)
            if not all(L is None for L in Ls): 
                nu_zls, nu_zus = zip(*[L.nu_zlu() for L in Ls if L is not None])
            else:
                nu_zls, nu_zus = [],[]

            # compute bounds
            # print(nu_hat_x.size(), sum(gamma).size(), l1.size(), sum(nu_zls).size())
            self.zl.append(nu_hat_x + sum(gamma) - epsilon*l1 + sum(nu_zls))
            self.zu.append(nu_hat_x + sum(gamma) + epsilon*l1 - sum(nu_zus))
        
        self.s = [torch.zeros_like(u) for l,u in zip(self.zl, self.zu)]

        for (s,l,u) in zip(self.s,self.zl, self.zu): 
            I_nonzero = (u != l).detach()
            if I_nonzero.data.sum() > 0: 
                s[I_nonzero] = u[I_nonzero]/(u[I_nonzero]-l[I_nonzero])

        
    def g(self, c):
        n = c.size(0)
        nu = [[]]*self.k
        nu[-1] = -c
        for i in range(self.k-2,-1,-1):
            nu[i] = batch(self.affine_transpose[i](unbatch(nu[i+1])),n)
            if i > 0:
                # avoid in place operation
                out = nu[i].clone()
                out[self.I_neg[i-1].unsqueeze(1)] = 0
                if not self.I_empty[i-1]:
                    if self.alpha_grad: 
                        out[self.I[i-1].unsqueeze(1)] = (self.s[i-1].unsqueeze(1).expand(*nu[i].size())[self.I[i-1].unsqueeze(1)] * 
                                                               nu[i][self.I[i-1].unsqueeze(1)])
                    else:
                        out[self.I[i-1].unsqueeze(1)] = ((self.s[i-1].unsqueeze(1).expand(*nu[i].size())[self.I[i-1].unsqueeze(1)] * 
                                                                                   torch.clamp(nu[i], min=0)[self.I[i-1].unsqueeze(1)])
                                                         + (self.s[i-1].detach().unsqueeze(1).expand(*nu[i].size())[self.I[i-1].unsqueeze(1)] * 
                                                                                   torch.clamp(nu[i], max=0)[self.I[i-1].unsqueeze(1)]))
                nu[i] = out

        f = (-sum(nu[i+1].matmul(self.biases[i].view(-1)) for i in range(self.k-1))
             -nu[0].matmul(self.X.view(self.X.size(0),-1).unsqueeze(2)).squeeze(2)
             -self.epsilon*nu[0].abs().sum(2)
             + sum((nu[i].clamp(min=0)*self.zl[i-1].unsqueeze(1)).matmul(self.I[i-1].type_as(self.X).unsqueeze(2)).squeeze(2) 
                    for i in range(1, self.k-1) if not self.I_empty[i-1]))

        return f

def robust_loss(net, epsilon, X, y, 
                size_average=True, **kwargs):
    num_classes = net[-1].out_features
    dual = DualNetBounds(net, X, epsilon, **kwargs)
    c = Variable(torch.eye(num_classes).type_as(X.data)[y.data].unsqueeze(1) - torch.eye(num_classes).type_as(X.data).unsqueeze(0))
    if X.is_cuda:
        c = c.cuda()
    f = -dual.g(c)
    err = (f.data.max(1)[1] != y.data)
    if size_average: 
        err = err.sum()/X.size(0)
    ce_loss = nn.CrossEntropyLoss(size_average=size_average)(f, y)
    return ce_loss, err