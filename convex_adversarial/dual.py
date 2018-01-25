import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
# import cvxpy as cp

from . import affine as Aff

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
        b = l.bias.unsqueeze(1).unsqueeze(2)
        k = int((n/(b.numel()))**0.5)
        return b.expand(b.numel(),k,k).contiguous().view(1,-1)
    else:
        raise ValueError("Full bias can't be formed for given layer.")

class DualNetBounds:    
    def __init__(self, net, x, epsilon):
        x = x.unsqueeze(0)
        # create network without bias terms
        self.layers = [l for l in net if isinstance(l, (nn.Linear, nn.Conv2d))]
        self.affine_transpose = [AffineTranspose(l) for l in self.layers]
        self.affine = [Affine(l) for l in self.layers]
        self.k = len(self.layers)+1

        # initialize affine layers with a forward pass
        _ = x.view(1,-1)
        for a in self.affine: 
            _ = a(_)
            
        self.biases = [full_bias(l, self.affine[i].out_features) 
                        for i,l in enumerate(self.layers)]
        
        gamma = [self.biases[0]]
        nu = []
        nu_hat = self.affine[0](x)
        nu_hat_ = self.affine[0](Variable(torch.eye(self.affine[0].in_features)).type_as(x))
        
        self.zl = [nu_hat.view(-1) + gamma[0].view(-1) - epsilon*(nu_hat_).abs().sum(0)]
        self.zu = [nu_hat.view(-1) + gamma[0].view(-1) + epsilon*(nu_hat_).abs().sum(0)]

        self.I = []
        self.I_empty = []
        self.I_neg = []
        self.I_pos = []
        self.x = x
        self.epsilon = epsilon
        

        for i in range(0,self.k-2):
            # compute sets and activation
            self.I_neg.append((self.zu[-1] <= 0).detach())
            self.I_pos.append((self.zl[-1] > 0).detach())
            self.I.append(((self.zu[-1] > 0) * (self.zl[-1] < 0)).detach())
            self.I_empty.append(self.I[-1].data.long().sum() == 0)
            d = self.I_pos[-1].type_as(x) + (self.zu[-1]/(self.zu[-1] - self.zl[-1]))*self.I[-1].type_as(x)
            
            # initialize new terms
            if not self.I_empty[-1]:
                out_features = self.affine[i+1].out_features
                subset_eye = Variable(x.data.new(self.I[-1].data.sum(), d.numel()).zero_())
                subset_eye.scatter_(1, Variable(self.I[-1].data.nonzero()), d[self.I[-1]][:,None])
                nu.append(self.affine[i+1](subset_eye.detach()).t())
            else:
                nu.append(Variable(torch.zeros(self.affine[i+1].out_features,1)).type_as(x))         
            gamma.append(self.biases[i+1])
            # propagate terms
            gamma[0] = self.affine[i+1](d * gamma[0])
            for j in range(1,i+1):
                gamma[j] = self.affine[i+1](d * gamma[j])
                nu[j-1] = self.affine[i+1]((d[:,None] * nu[j-1]).t()).t()
            nu_hat = self.affine[i+1](d*nu_hat)
            nu_hat_ = self.affine[i+1](d*nu_hat_)
            
            # compute bounds
            self.zl.append(nu_hat.view(-1) + sum(gamma).view(-1) - epsilon*nu_hat_.abs().sum(0) + 
                           sum([(self.zl[j][self.I[j]] * (-nu[j]).clamp(min=0)).sum(1) 
                                for j in range(i+1) if not self.I_empty[j]]))
            self.zu.append(nu_hat.view(-1) + sum(gamma).view(-1) + epsilon*nu_hat_.abs().sum(0) - 
                           sum([(self.zl[j][self.I[j]] * nu[j].clamp(min=0)).sum(1) 
                                for j in range(i+1) if not self.I_empty[j]]))

        self.s = [(u/(u-l)) for l,u in zip(self.zl, self.zu)]
        
            
    def g(self, c):
        # print("serialized start")
        nu = [[]]*self.k
        nu[-1] = -c
        for i in range(self.k-2,-1,-1):
            nu[i] = self.affine_transpose[i](nu[i+1])
            if i > 0:
                nu[i][self.I_neg[i-1].expand_as(nu[i])] = 0
                if not self.I_empty[i-1]:
                    #print(self.s[i-1][self.I[i-1]].repeat(nu[i].size(0)))
                    # print((self.s[i-1][self.I[i-1]].repeat(nu[i].size(0)) * 
                    #                                        nu[i][self.I[i-1].expand_as(nu[i])]))
                    #assert False
                    nu[i][self.I[i-1].expand_as(nu[i])] = (self.s[i-1][self.I[i-1]].repeat(nu[i].size(0)) * 
                                                           nu[i][self.I[i-1].expand_as(nu[i])])
        
        f = (-sum(nu[i+1].mm(self.biases[i].view(-1,1))[:,0] for i in range(self.k-1)) 
             - nu[0].mm(self.x.view(-1,1))[:,0]
             - self.epsilon*nu[0].abs().sum(1)
             + sum((nu[i].clamp(min=0)*self.zl[i-1][None,:])[self.I[i-1].expand_as(nu[i])].view(nu[i].size(0),-1).sum(1)
                   for i in range(1,self.k-1) if not self.I_empty[i-1]))
        # for n in nu:
        #     print(n)
        # print(f.data)
        return f

class DualNetBoundsBatch: 
    def __init__(self, net, X, epsilon, alpha_grad, scatter_grad,
                 l1_proj=None):
        self.layers = [l for l in net if isinstance(l, (nn.Linear, nn.Conv2d))]
        self.affine_transpose = [AffineTranspose(l) for l in self.layers]
        self.affine = [Affine(l) for l in self.layers]
        self.k = len(self.layers)+1

        # initialize affine layers with a forward pass
        _ = X[0].view(1,-1)
        for a in self.affine: 
            _ = a(_)
            
        self.biases = [full_bias(l, self.affine[i].out_features) 
                        for i,l in enumerate(self.layers)]
        
        gamma = [self.biases[0]]
        nu = []
        nu_hat = self.affine[0](X)

        if l1_proj is None: 
            eye = Variable(torch.eye(self.affine[0].in_features)).type_as(X)
        else: 
            if not isinstance(l1_proj, int): 
                raise ValueError('l1_proj must be an integer')
            # Use Cauchy random projections
            eye = Variable(torch.zeros(l1_proj, 
                                       self.affine[0].in_features
                                      ).cauchy_()).type_as(X)
        nu_hat_ = self.affine[0](eye).unsqueeze(0)

        if l1_proj is None: 
            l1 = (nu_hat_).abs().sum(1)
        else: 
            # use an approximation
            l1 = torch.median(nu_hat_.abs(), 1)[0]

        self.zl = [nu_hat + gamma[0] - epsilon*l1]
        self.zu = [nu_hat + gamma[0] + epsilon*l1]

        self.I = []
        self.I_empty = []
        self.I_neg = []
        self.I_pos = []
        self.X = X
        self.epsilon = epsilon
        I_collapse = []
        I_ind = []

        # set flags
        self.scatter_grad = scatter_grad
        self.alpha_grad = alpha_grad

        for i in range(0,self.k-2):
            # compute sets and activation
            self.I_neg.append((self.zu[-1] <= 0).detach())
            self.I_pos.append((self.zl[-1] > 0).detach())
            self.I.append(((self.zu[-1] > 0) * (self.zl[-1] < 0)).detach())
            self.I_empty.append(self.I[-1].data.long().sum() == 0)
            d = self.I_pos[-1].type_as(X) + (self.zu[-1]/(self.zu[-1] - self.zl[-1]))*self.I[-1].type_as(X)
            d[((self.zu[-1].data-self.zl[-1].data) == 0)*self.I[-1].data] = 0

            # indices of [example idx, origin crossing feature idx]
            I_ind.append(Variable(self.I[-1].data.nonzero()))
            
            # initialize new terms
            if not self.I_empty[-1]:
                out_features = self.affine[i+1].out_features

                subset_eye = Variable(X.data.new(self.I[-1].data.sum(), d.size(1)).zero_())
                subset_eye.scatter_(1, I_ind[-1][:,1,None], d[self.I[-1]][:,None])

                if not scatter_grad: 
                    subset_eye = subset_eye.detach()
                nu.append(self.affine[i+1](subset_eye).t())

                # create a matrix that collapses the minibatch of origin-crossing indices 
                # back to the sum of each minibatch
                I_collapse.append(Variable(X.data.new(I_ind[-1].size(0), X.size(0)).zero_()))
                I_collapse[-1].scatter_(1, I_ind[-1][:,0][:,None], 1)
            else:
                nu.append(Variable(torch.zeros(1, self.affine[i+1].out_features)).type_as(X))         
                I_collapse.append(None)
            gamma.append(self.biases[i+1])
            # propagate terms
            gamma[0] = self.affine[i+1](d * gamma[0])
            for j in range(1,i+1):
                gamma[j] = self.affine[i+1](d * gamma[j])
                if not self.I_empty[j-1]: 
                    nu[j-1] = self.affine[i+1]((d[I_ind[j-1][:,0]].t() * nu[j-1]).t()).t()

            nu_hat = self.affine[i+1](d*nu_hat)
            nu_hat_ = self.affine[i+1]((d.unsqueeze(1)*nu_hat_).view(-1, d.size(1))).view(d.size(0), nu_hat_.size(1), -1)
            
            if l1_proj is None: 
                l1 = (nu_hat_).abs().sum(1)
            else: 
                # use an approximation
                l1 = torch.median(nu_hat_.abs(), 1)[0]

            # compute bounds
            self.zl.append(nu_hat + sum(gamma) - epsilon*l1 + 
                           sum([(self.zl[j][self.I[j]] * (-nu[j]).clamp(min=0)).mm(I_collapse[j]).t()
                                for j in range(i+1) if not self.I_empty[j]]))
            self.zu.append(nu_hat + sum(gamma) + epsilon*l1 - 
                           sum([(self.zl[j][self.I[j]] * nu[j].clamp(min=0)).mm(I_collapse[j]).t()
                                for j in range(i+1) if not self.I_empty[j]]))
        
        self.s = [(u/(u-l)) for l,u in zip(self.zl, self.zu)]

        for (s,l,u) in zip(self.s,self.zl, self.zu): 
            s[((u-l) == 0).detach()] = 0

        
    def g(self, c):
        # print("minibatched start")
        nu = [[]]*self.k
        nu[-1] = -c
        for i in range(self.k-2,-1,-1):
            nu[i] = self.affine_transpose[i](nu[i+1].view(-1, nu[i+1].size(2))).view(c.size(0), c.size(1), -1)
            if i > 0:
                nu[i][self.I_neg[i-1].unsqueeze(1)] = 0
                if not self.I_empty[i-1]:
                    out = nu[i].clone()
                    # avoid in place operation
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

def robust_loss(net, epsilon, X, y):
    num_classes = net[-1].out_features
    ce_loss = 0.0
    err = 0.0
    # not batched
    for i in range(X.size(0)):
        dual = DualNetBounds(net, X[i], epsilon)
        c = Variable((torch.eye(num_classes)[:,y.data[i]] - torch.eye(num_classes)))
        if X.is_cuda:
            c = c.cuda()

        f = -dual.g(c)
        
        err += (f.data.max(0)[1] != y.data[i]).float()[0]
        #hinge_loss += ((Variable(1-torch.eye(num_classes)[:,y[i]]) + f).max()).clamp(min=0)
        ce_loss += nn.CrossEntropyLoss()(f[None,:], y[i:i+1])
        
    return ce_loss/X.size(0), err/X.size(0)


def robust_loss_batch(net, epsilon, X, y, alpha_grad, scatter_grad, 
                      l1_proj=None):
    num_classes = net[-1].out_features
    ce_loss = 0.0
    err = 0.0
    # batched
    dual = DualNetBoundsBatch(net, X, epsilon, alpha_grad, scatter_grad, 
                              l1_proj=l1_proj)
    c = Variable(torch.eye(num_classes).type_as(X.data)[y.data].unsqueeze(1) - torch.eye(num_classes).type_as(X.data).unsqueeze(0))
    if X.is_cuda:
        c = c.cuda()
    f = -dual.g(c)
    err = (f.data.max(1)[1] != y.data).sum()/X.size(0)
    ce_loss = nn.CrossEntropyLoss()(f, y)
    return ce_loss, err