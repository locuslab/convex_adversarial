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
        if nu is None: 
            l1 = self.nu_1[-1].abs().sum(1)
            return (self.nu_x[-1] - self.epsilon*l1, 
                    self.nu_x[-1] + self.epsilon*l1)
        else: 
            nu = nu.view(nu.size(0), nu.size(1), -1)
            nu_x = nu.matmul(self.nu_x[0].view(self.nu_x[0].size(0),-1).unsqueeze(2)).squeeze(2)
            l1 = self.epsilon*nu.abs().sum(2)
            return -nu_x - l1

class DualLinear(): 
    def __init__(self, layer, out_features): 
        if not isinstance(layer, nn.Linear):
            raise ValueError("Expected nn.Linear input.")
        self.layer = layer
        self.bias = [Aff.full_bias(layer, out_features)]

    def apply(self, dual_layer): 
        self.bias.append(dual_layer.affine(*self.bias))

    def fval(self, nu=None, nu_prev=None): 
        if nu is None: 
            return self.bias[-1], self.bias[-1]
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
        self.bias = [Aff.full_bias(layer, out_features).contiguous()]

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
        if isinstance(dual_layer, (DualReLU, ApplyF)): 
            self.nus.append(dual_layer.affine(*self.nus, I_ind=self.I_ind))
        else: 
            self.nus.append(dual_layer.affine(*self.nus))

    def fval(self, nu=None, nu_prev=None): 
        if self.I_empty: 
            return True
        if nu_prev is None:
            nu = self.nus[-1]
            nu = nu.view(nu.size(0), -1)
            zlI = self.zl[self.I]
            zl = (zlI * (-nu.t()).clamp(min=0)).mm(self.I_collapse).t().contiguous()
            zu = -(zlI * nu.t().clamp(min=0)).mm(self.I_collapse).t().contiguous()
            
            zl = zl.view(-1, *(self.nus[-1].size()[1:]))
            zu = zu.view(-1, *(self.nus[-1].size()[1:]))
            return zl,zu
        else: 
            n = nu_prev.size(0)
            nu = nu_prev.view(n, nu_prev.size(1), -1)
            zl = self.zl.view(n, -1)
            I = self.I.view(n, -1)
            return (nu.clamp(min=0)*zl.unsqueeze(1)).matmul(I.type_as(nu).unsqueeze(2)).squeeze(2)

    def affine(self, *xs, I_ind=None): 
        x = xs[-1]
        # if x.dim() == 2: 
        #     x = convert4to2(x)
        d = self.d 
        if x.dim() > d.dim():
            d = d.unsqueeze(1)

        if I_ind is not None: 
            return d[I_ind[:,0]]*x
        else:
            return d*x

    def affine_transpose(self, *xs): 
        return self.affine(*xs)

class ApplyF():
    def __init__(self, f): 
        self.f = f 
    def affine(self, *xs, **kwargs):
        return self.f(*xs, **kwargs)

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
        _ = X[0:1]
        nf = [_.size()]
        for l in net: 
            _ = l(_)
            nf.append(_.size())
        dual_net = [InfBall(X, epsilon)]
        # i = 0
        for i,(in_f,out_f,layer) in enumerate(zip(nf[:-1], nf[1:], net)): 
            if isinstance(layer, nn.Linear): 
                dual_layer = DualLinear(layer, out_f)
            elif isinstance(layer, nn.Conv2d): 
                dual_layer = DualConv2d(layer, out_f)
            elif isinstance(layer, nn.ReLU): 
                zl, zu = zip(*[l.fval() for l in dual_net])
                # print(dual_net[-2], zl[-2])
                #     for l in dual_net:
                #         print(l.fval()[0])
                #     assert False 
                # i += 1
                zl, zu = sum(zl), sum(zu)
                # print(zl.data.sum(), zu.data.sum(), i)
                d = (zl > 0).detach().type_as(X)
                I = ((zu > 0) * (zl < 0)).detach()
                if I.data.sum() > 0:
                    d[I] += zu[I]/(zu[I] - zl[I])
                # print(d.data.norm(), I.data.sum())
                dual_layer = DualReLU(I, d, zl)
            elif 'Flatten' in (str(layer.__class__.__name__)): 
                dual_layer = DualReshape(in_f, out_f)
            else:
                print(layer)
                raise ValueError

            # skip last layer
            if i < len(net)-1: 
                for l in dual_net: 
                    l.apply(dual_layer)
                dual_net.append(dual_layer)
            else: 
                self.last_layer = dual_layer
        self.dual_net = dual_net
        return 

        n = X.size(0)
        self.layers = [l for l in net if isinstance(l, (nn.Linear, nn.Conv2d, Dense))]

        # self.affine_transpose = [Aff.toAffineTranspose(l) for l in self.layers]
        self.affine_transpose = Aff.transpose_all(self.layers)
        self.affine = [Aff.toAffine(l) for l in self.layers]

        self.k = len(self.layers)+1

        # initialize affine layers with a forward pass
        _ = [X[0].view(1,-1)]
        for a in self.affine: 
            _.append(a(*_))

        self.biases = [Aff.full_bias(l, self.affine[i].out_features) 
                        for i,l in enumerate(self.layers)]
        
        forward_gamma = [ForwardPass(self.biases[0])]
        gamma = [self.biases[0]]

        forward_x = ForwardPass(X)
        nu_hat_x = forward_x.apply(self.affine[0])

        L0 = select_L(X, l1_proj, m, l1_eps, self.affine[0], l1_type=l1_type,
                      threshold=self.affine[0].in_features)
        l1 = L0.l1_norm()

        self.zl = [nu_hat_x + gamma[0] - epsilon*l1]
        self.zu = [nu_hat_x + gamma[0] + epsilon*l1]
        tmp22 = self.zl[0].data

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
            d2 = d.data
            # if i == 1: 
            #     l_ = [nu_hat_x + epsilon*l1] + list(gamma) + list(nu_zus)
            #     print([l0_.data.norm() for l0_ in l_])
            #     # print(self.zu[-1].norm())
            #     assert False
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
            nu0_2 = (Ls[0].nus[0]).data
            I2 = self.I[-1].data
            zl2 = self.zl[-1].data

            forward_gamma.append(ForwardPass(self.biases[i+1]))
            gamma.append(self.biases[i+1])
            # propagate bias terms
            for j in range(0,i+1):
                gamma[j] = forward_gamma[j].add_and_apply(d * gamma[j], self.affine[i+1])
            
            nu_hat_x = forward_x.add_and_apply(d*nu_hat_x, self.affine[i+1])
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
            # if i == 1:
            #     print(nu_hat_x - epsilon*l1, gamma, nu_zls)
            #     assert False
            self.zl.append(nu_hat_x + sum(gamma) - epsilon*l1 + sum(nu_zls))
            self.zu.append(nu_hat_x + sum(gamma) + epsilon*l1 - sum(nu_zus))
        
        self.s = [torch.zeros_like(u) for l,u in zip(self.zl, self.zu)]
        for (s,l,u) in zip(self.s,self.zl, self.zu): 
            I_nonzero = (u != l).detach()
            if I_nonzero.data.sum() > 0: 
                s[I_nonzero] = u[I_nonzero]/(u[I_nonzero]-l[I_nonzero])
        # print("*"*80)
        # print((tmp11.view_as(tmp22)-tmp22).norm())
        # print((dual_layer.d.data.view_as(d2)-d2).norm())
        # print((d1[I1]-d2[I2]).norm())
        # print((dual_layer.zl.data.view_as(zl2)-zl2).norm())
        # print((dual_layer.I.data.view_as(I2)-I2).float().norm())
        # print((dual_layer.nus[0].data-nu0_2).norm())
        # print(nu0_1.norm(), nu0_2.norm())
        # print(I.data.nonzero(), I2)
        # print((tmp1.view_as(tmp2)-tmp2).norm())
        # print([g.data.sum() for g in nu_zls])
        # for zl,zu in zip(self.zl,self.zu): 
        #     print(zl.data.sum(), zu.data.sum())
        # assert False
        # print(len(self.zl))
        
    def g(self, c):
        nu = [-c]
        nu.append(self.last_layer.affine_transpose(nu[0]))
        for l in reversed(self.dual_net[1:]): 
            nu.append(l.affine_transpose(nu[-1]))
            # print(nu[-1].size())

        dual_net = self.dual_net + [self.last_layer]
        
        nu_ = nu

        nu.append(None)
        nu = list(reversed(nu))
        # [None, nu1, ..., nuk=c]

        out = sum(l.fval(nu=n, nu_prev=nprev) 
            for l,nprev,n in zip(dual_net, nu[:-1],nu[1:]))
        return out


        # nu = list(reversed(nu[1:]))
        # out = sum(l.fval(nu=n) for l,n in zip(self.dual_net, nu))
        # print(out)
        # print(nu[0])
        # nu_ = [nu[0], nu[1], nu[3], nu[5], nu[7], nu[9]]
        # assert False
        # if self.last_layer is not None: 
        #     nu = batch(self.last_layer.affine_transpose(unbatch(c)), c.size(0)).transpose(-1,-2)
        #     def mm_nu(*xs, I_ind=None): 
        #         squeeze = False
        #         x = xs[-1]
        #         if x.dim() < nu.dim():
        #             squeeze = True
        #             x = x.unsqueeze(1)
        #         if I_ind is not None: 
        #             nu0 = nu[I_ind[:,0]]
        #         else: 
        #             nu0  = nu
        #         if squeeze: 
        #             return x.bmm(nu0).squeeze(1)
        #         else:
        #             return x.bmm(nu0)
        #     F = ApplyF(mm_nu)
        #     for l in self.dual_net: 
        #         l.apply(F)
        #     self.dual_net.append(self.last_layer)
        #     self.last_layer = None


        n = c.size(0)
        nu = [[]]*self.k
        nu[-1] = -unbatch(c)
        for i in range(self.k-2,-1,-1):
            nu[i] = self.affine_transpose[i](*list(reversed(nu[i+1:])))
            if i > 0:
                # avoid in place operation
                out = batch(nu[i],n).clone()
                out[self.I_neg[i-1].unsqueeze(1)] = 0
                if not self.I_empty[i-1]:
                    idx = self.I[i-1].unsqueeze(1)
                    nu_i = batch(nu[i],n)
                    if self.alpha_grad: 
                        out[idx] = (self.s[i-1].unsqueeze(1).expand(*nu_i.size())[self.I[i-1].unsqueeze(1)] * 
                                                               nu_i[self.I[i-1].unsqueeze(1)])
                    else:
                        out[idx] = ((self.s[i-1].unsqueeze(1).expand(*nu_i.size())[idx] * 
                                                                                   torch.clamp(nu_i, min=0)[idx])
                                                         + (self.s[i-1].detach().unsqueeze(1).expand(*nu_i.size())[idx] * 
                                                                                   torch.clamp(nu_i, max=0)[idx]))
                nu[i] = unbatch(out)

        nu = [batch(nu0,n) for nu0 in nu]
        f = (-sum(nu[i+1].matmul(self.biases[i].view(-1)) for i in range(self.k-1))
             -nu[0].matmul(self.X.view(self.X.size(0),-1).unsqueeze(2)).squeeze(2)
             -self.epsilon*nu[0].abs().sum(2)
             + sum((nu[i].clamp(min=0)*self.zl[i-1].unsqueeze(1)).matmul(self.I[i-1].type_as(self.X).unsqueeze(2)).squeeze(2) 
                    for i in range(1, self.k-1) if not self.I_empty[i-1]))
        # print(-self.epsilon*nu[0].abs().sum(2)-nu[0].matmul(self.X.view(self.X.size(0),-1).unsqueeze(2)).squeeze(2))
        # for i in range(len(nu_)): 
            # print(nu_[i].size(), nu[i].size())
            # print((nu[i].data-nu_[i].data).norm())
        # print(nu[0].abs().sum(2))
        # print(nu_[0].abs().sum(2))
        # print(nu[0])
        # print(f)
        # assert False
        # print((nu_[0] - nu[5]).data.norm())
        # print((nu_[2] - nu[4]).data.norm())
        # print((nu_[4] - nu[3]).data.norm())
        # print((nu_[6] - nu[2]).data.norm())
        # print((nu_[8] - nu[1]).data.norm())
        # print((nu_[9] - nu[0]).data.norm())
        # self.dual_net.append(self.last_layer)
        # print((self.dual_net[0].fval(nu=nu_[9])-nu[0].matmul(self.X.view
        #                     (self.X.size(0),-1).unsqueeze(2)).squeeze(2)
        #                 -self.epsilon*nu[0].abs().sum(2)).data.norm())
        # for i in range(self.k-1): 
        #     print(1+2*i, 8-2*i)
        #     print((self.dual_net[1+2*i].fval(nu_[8-2*i]) - 
        #                       nu[1+i].matmul(self.biases[0+i].view
        #                         (-1))).data.norm())

        # for i in range(1, self.k-1): 
        #     print(2*i, 9-2*i)
        #     print((self.dual_net[2*i].fval(nu=nu_[9 - 2*i], nu_prev=nu_
        #         [10-2*i])-(nu[i].clamp
        #                     (min=0)*self.zl[i-1].unsqueeze(1)).matmul(self.I
        #                     [i-1].type_as(self.X).unsqueeze(2)).squeeze(2)
        #                     ).data.norm())

        # print(len(nu_), len(self.dual_net))
        # print("*"*80)
        # print()
        # # print(nu_[1].size(), nu_[2].size(), nu[-2].size())
        # # print(nu_[0])
        # # print(nu[-1])
        # assert False
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