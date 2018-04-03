import torch
from torch.autograd import Variable
from abc import ABC, abstractmethod

import numpy as np
import torch.nn as nn

def batch(A, n): 
    return A.view(n, -1, *A.size()[1:])
def unbatch(A): 
    return A.view(-1, *A.size()[2:])


class Norm(ABC): 
    @abstractmethod
    def input(self): 
        raise NotImplemented

    @abstractmethod
    def add_layer(self):
        raise NotImplemented

    @abstractmethod
    def skip(self):
        raise NotImplemented

    @abstractmethod
    def apply(self):
        raise NotImplemented

    @abstractmethod
    def l1_norm(self): 
        raise NotImplemented

    @abstractmethod
    def nu_zl(self):
        raise NotImplemented

    @abstractmethod
    def nu_zu(self):
        raise NotImplemented

class L1(Norm): 
    def __init__(self, X): 
        self.X = X
        self.I_collapse = []
        self.I_ind = []
        self.nu = []

    def input(self, in_features): 
        return Variable(torch.eye(in_features)).type_as(self.X)

    def add_layer(self, W, I, d, scatter_grad): 
        # indices of [example idx, origin crossing feature idx]
        self.I_ind.append(Variable(I[-1].data.nonzero()))
        subset_eye = Variable(self.X.data.new(I[-1].data.sum(), d.size(1)).zero_())
        subset_eye.scatter_(1, self.I_ind[-1][:,1,None], d[I[-1]][:,None])

        # create a matrix that collapses the minibatch of origin-crossing indices 
        # back to the sum of each minibatch
        self.I_collapse.append(Variable(self.X.data.new(self.I_ind[-1].size(0),self.X.size(0)).zero_()))
        self.I_collapse[-1].scatter_(1, self.I_ind[-1][:,0][:,None], 1)

        if not scatter_grad: 
            subset_eye = subset_eye.detach()

        self.nu.append(W(subset_eye))

    def skip(self): 
        self.nu.append(None)         
        self.I_collapse.append(None)
        self.I_ind.append(None)


    def apply(self, W, d):
        """ Scale all variables by d and pass through layer W """
        for j in range(len(self.nu)): 
            if self.nu[j] is not None: 
                self.nu[j] = W(d[self.I_ind[j][:,0]] * self.nu[j])


    def l1_norm(self, nu_hat_1): 
        return (nu_hat_1).abs().sum(1)

    def nu_zlu(self, zl, I):
        return self.nu_zl(zl,I), self.nu_zu(zl,I)
        # terms = [(zl[j][I[j]], self.nu[j].t().clamp(min=0))
        #                         for j in range(len(zl)) if not self.nu[j] is None]
        # nu_zl = sum((t0*(-t1)).mm(self.I_collapse[j]).t() for t0,t1 in terms)
        # nu_zu = sum((t0*t1) for t0,t1 in terms)
        # return nu_zl, nu_zu


    def nu_zl(self, zl, I):
        return sum([(zl[j][I[j]] * (-self.nu[j].t()).clamp(min=0)).mm(self.I_collapse[j]).t()
                                for j in range(len(zl)) if not self.nu[j] is None])

    def nu_zu(self, zl, I):
        return sum([(zl[j][I[j]] * self.nu[j].t().clamp(min=0)).mm(self.I_collapse[j]).t()
                                for j in range(len(zl)) if not self.nu[j] is None])

class L1_Cauchy(Norm):
    def __init__(self, X, k, m, l1_eps): 
        self.X = X
        self.k = k
        self.m = m
        self.epsilon = l1_eps
        self.nu = []
        self.nu_one = []

    def input(self, in_features): 
        return Variable(torch.zeros(self.k*self.m, in_features).cauchy_()).type_as(self.X)

    def add_layer(self, W, I, d, scatter_grad, zl=[]):
        B = Variable(torch.zeros(1, self.k*self.m, d.size(1)).cauchy_()).type_as(self.X)
        ones = Variable(torch.ones(1, d.size(1))).type_as(self.X)

        if  (~I[-1].data).sum() > 0: 
            B[:,:,(~I[-1].data).nonzero().squeeze(1)] = 0
            ones[:, (~I[-1].data).nonzero().squeeze(1)] = 0
        B = zl[-1].unsqueeze(1)*B
        ones = zl[-1]*ones

        self.nu.append(W(unbatch(B)))
        self.nu_one.append(W(ones))

    def skip(self):
        self.nu.append(None)
        self.nu_one.append(None)

    def apply(self, W, d):
        for j in range(len(self.nu)): 
            if self.nu[j] is not None: 
                self.nu[j] = W(unbatch(d.unsqueeze(1) * batch(self.nu[j],
                   self.X.size(0))))
                self.nu_one[j] = W(d * self.nu_one[j])

    def nu_zlu(self, zl): 
        terms = [(self.l1_norm(batch(n, self.X.size(0))),no) 
                     for n,no in zip(self.nu, self.nu_one) if n is not None]
        nu_zl = sum((-n + no)/2 for n,no in terms)
        nu_zu = sum((-n - no)/2 for n,no in terms)
        return nu_zl,nu_zu


    def nu_zl(self, zl):
        return sum([(-self.l1_norm(batch(n, self.X.size(0))) + no)/2 
                     for n,no in zip(self.nu, self.nu_one) if n is not None])

    def nu_zu(self, zl):
        return sum([(-self.l1_norm(batch(n, self.X.size(0))) - no)/2 
                     for n,no in zip(self.nu, self.nu_one) if n is not None])

class L1_median(L1_Cauchy): 
    def l1_norm(self, nu_hat_1): 
        # return torch.median(nu_hat_1.abs(), 1)[0]/(1-self.epsilon)
        nu_hat_1 = nu_hat_1.view(-1, self.m, self.k, nu_hat_1.size(2))
        return torch.max(torch.median(nu_hat_1.abs(), 2)[0], 1)[0]/(1-self.epsilon)
def GR(epsilon): 
    return (epsilon**2)/(-0.5*np.log(1+(2/np.pi*np.log(1+epsilon))**2) 
                        + 2/np.pi*np.arctan(2/np.pi*np.log(1+epsilon))*np.log(1+epsilon))

def GL(epsilon): 
    return (epsilon**2)/(-0.5*np.log(1+(2/np.pi*np.log(1-epsilon))**2) 
                        + 2/np.pi*np.arctan(2/np.pi*np.log(1-epsilon))*np.log(1-epsilon))

def p_upper(epsilon, k): 
    return np.exp(-k*(epsilon**2)/GR(epsilon))

def p_lower(epsilon, k): 
    return np.exp(-k*(epsilon**2)/GL(epsilon))

import time
def epsilon_from_model(model, X, k, delta, m): 
    if k is None or delta is None or m is None: 
        raise ValueError("k, delta, and m must not be None. ")
    X = X[0].unsqueeze(0)
    out_features = []
    for l in model: 
        X = l(X)
        if isinstance(l, (nn.Linear, nn.Conv2d)): 
            out_features.append(X.numel())

    num_est = sum(n for n in out_features[:-1])

    num_est += sum(n*i for i,n in enumerate(out_features[:-1]))

    sub_delta = (delta/num_est)**(1/m)
    l1_eps = get_epsilon(sub_delta, k)

    if l1_eps > 1: 
        raise ValueError('Delta too large / k too small to get probabilistic bound')
    return l1_eps

def get_epsilon(delta, k, alpha=1e-2): 
    """ Determine the epsilon for which the estimate is accurate
    with probability >(1-delta) and k projection dimensions. """
    epsilon = 0.001
    # probability of incorrect bound 
    start_time = time.time()
    p_max = max(p_upper(epsilon, k), p_lower(epsilon,k))
    while p_max > delta: 
        epsilon *= (1+alpha)
        p_max = max(p_upper(epsilon, k), p_lower(epsilon,k))
    if epsilon > 1: 
        raise ValueError('Delta too large / k too small to get probabilistic bound (epsilon > 1)')
    # print(time.time()-start_time)
    return epsilon

class L1_geometric(L1_Cauchy): 
    def add_layer(self, W, I, d, scatter_grad, zl=[]):
        return super(L1_geometric, self).add_layer(W, I, d, scatter_grad, zl=zl)

    def l1_norm(self, nu_hat_1): 
        batch_size = self.X.size(0)
        nu_hat_1 = nu_hat_1.view(-1, self.m, self.k, nu_hat_1.size(2))
        # original = (torch.exp((torch.log(nu_hat_1[0:1].abs())/k).sum(1)))/(1-self.epsilon)
        new = (torch.max(torch.exp((torch.log(nu_hat_1.abs())/self.k).sum(2))/
            (1-self.epsilon), 1)[0])
        return new
