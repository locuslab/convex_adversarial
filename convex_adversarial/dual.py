import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import cvxpy as cp

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
            self.I_neg.append(self.zu[-1] <= 0)
            self.I_pos.append(self.zl[-1] > 0)
            self.I.append((self.zu[-1] > 0) * (self.zl[-1] < 0))
            self.I_empty.append(self.I[-1].data.long().sum() == 0)
            d = self.I_pos[-1].type_as(x) + (self.zu[-1]/(self.zu[-1] - self.zl[-1]))*self.I[-1].type_as(x)
            
            # initialize new terms
            if not self.I_empty[-1]:
                out_features = self.affine[i+1].out_features
                subset_eye = x.data.new(self.I[-1].data.sum(), d.numel()).zero_()
                subset_eye.scatter_(1, self.I[-1].data.nonzero(), d[self.I[-1]].data[:,None])
                nu.append(self.affine[i+1](Variable(subset_eye)).t())
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

        self.alpha_ind = [0] + np.cumsum([i.data.sum() for i in self.I]).tolist()
        self.s = [(u/(u-l)) for l,u in zip(self.zl, self.zu)]
        
            
    def g(self, c, alpha):
        nu = [[]]*self.k
        nu[-1] = -c
        for i in range(self.k-2,-1,-1):
            nu[i] = self.affine_transpose[i](nu[i+1])
            if i > 0:
                nu[i][self.I_neg[i-1].expand_as(nu[i])] = 0
                if not self.I_empty[i-1]:
                    nu[i][self.I[i-1].expand_as(nu[i])] = (self.s[i-1][self.I[i-1]].repeat(nu[i].size(0)) * 
                                                          torch.clamp(nu[i][self.I[i-1].expand_as(nu[i])],min=0) + 
                                                          torch.clamp(nu[i][self.I[i-1].expand_as(nu[i])],max=0) *
                                                          alpha[:,self.alpha_ind[i-1]:self.alpha_ind[i]].contiguous().view(-1))
        
        f = (-sum(nu[i+1].mm(self.biases[i].view(-1,1))[:,0] for i in range(self.k-1)) 
             - nu[0].mm(self.x.view(-1,1))[:,0]
             - self.epsilon*nu[0].abs().sum(1)
             + sum((nu[i].clamp(min=0)*self.zl[i-1][None,:])[self.I[i-1].expand_as(nu[i])].view(nu[i].size(0),-1).sum(1)
                   for i in range(1,self.k-1) if not self.I_empty[i-1]))
        return f
    
    def opt_alpha_default(self, c):
        all_alpha = [self.s[i][self.I[i]] for i in range(self.k-2) if not self.I_empty[i]]
        if len(all_alpha) > 0:
            alpha_ = torch.cat(all_alpha).data
            alpha = Variable(alpha_[None,:].expand(c.size(0),self.num_alpha), requires_grad=True)
            return alpha
        else:
            return None
    
    def opt_alpha_fgs(self, c):
        alpha = self.opt_alpha_default(c)
        f = self.g(c, alpha)
        (-f.sum()).backward(retain_graph=True)
        alpha = Variable((alpha.grad.data <0).float())
        return alpha
    
    def opt_alpha_adam(self, c, steps=20):
        alpha = self.opt_alpha_default(c)
        opt = optim.Adam([alpha], lr=0.5)
        for i in range(steps):
            f = self.g(c, alpha)
            opt.zero_grad()
            (-f.sum()).backward(retain_graph=True)
            opt.step()
            alpha.data.clamp_(min=0, max=1)
        return Variable(alpha.data)
    
    @property
    def num_alpha(self):
        return self.alpha_ind[-1]


def robust_loss(net, epsilon, X, y, alpha='default'):
    num_classes = net[-1].out_features
    ce_loss = 0.0
    err = 0.0
    
    for i in range(X.size(0)):
        dual = DualNetBounds(net, X[i], epsilon)
        c = Variable((torch.eye(num_classes)[:,y.data[i]] - torch.eye(num_classes)))
        if X.is_cuda:
            c = c.cuda()

        if alpha == 'default': 
            a = dual.opt_alpha_default(c)
        elif alpha == 'fgs': 
            a = dual.opt_alpha_fgs(c)
        elif alpha == 'Adam': 
            a = dual.opt_alpha_adam(c)
        else:
            raise ValueError('Unknown alpha type, must be ["default", "fgs", "Adam"]')

        f = -dual.g(c, a)
        
        err += (f.data.max(0)[1] != y.data[i]).float()
        #hinge_loss += ((Variable(1-torch.eye(num_classes)[:,y[i]]) + f).max()).clamp(min=0)
        ce_loss += nn.CrossEntropyLoss()(f[None,:], y[i:i+1])
        
    return ce_loss/X.size(0), err/X.size(0)