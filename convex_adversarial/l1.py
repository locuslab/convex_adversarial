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

class L1(): 
    def __init__(self, X, W, I=None, d=None, scatter_grad=None, zl=None):
        self.X = X
        kwargs = [I, d, scatter_grad, zl]
        if not (all(kwarg is None for kwarg in kwargs) or 
            all(kwarg is not None for kwarg in kwargs)):
            print(kwargs)
            raise ValueError('Must either specify all keyword arguments or none')

        self.is_input = all(kwarg is None for kwarg in kwargs)
        if self.is_input: 
            self.nu = W(Variable(torch.eye(W.in_features)).type_as(self.X)).unsqueeze(0)
        else:
            self.I_ind = Variable(I.data.nonzero())
            self.nu = Variable(X.data.new(I.data.sum(), d.size(1)).zero_())
            self.nu.scatter_(1, self.I_ind[:,1,None], d[I][:,None])

            if not scatter_grad:
                self.nu = self.nu.detach()

            self.nu = W(self.nu)
            self.I = I
            self.zl = zl
            self.I_collapse = Variable(X.data.new(self.I_ind.size(0),X.size(0)).zero_())
            self.I_collapse.scatter_(1, self.I_ind[:,0][:,None], 1)

    
    def apply(self, W, d):
        if self.is_input: 
            n = self.X.size(0)
            self.nu = batch(W(unbatch(d.unsqueeze(1)*self.nu)), n)
        else:
            # print(d[self.I_ind[:,0]].size(), self.nu.size())
            self.nu = W(d[self.I_ind[:,0]] * self.nu)
        # self.nu = W(unbatch(d.unsqueeze(1) * batch(self.nu, self.X.size(0))))

    def nu_zlu(self): 
        nu_zl = (self.zl[self.I] * (-self.nu.t()).clamp(min=0)).mm(self.I_collapse).t()
        nu_zu = (self.zl[self.I] * self.nu.t().clamp(min=0)).mm(self.I_collapse).t()
        return nu_zl,nu_zu

    def l1_norm(self): 
        return self.nu.abs().sum(1)

class L1_Cauchy(): 
    def __init__(self, X, k, m, l1_eps, W, I=None, d=None, scatter_grad=None, zl=None): 
        kwargs = [I, d, scatter_grad]
        if not (all(kwarg is None for kwarg in kwargs) or 
            all(kwarg is not None for kwarg in kwargs)):
            raise ValueError('Must either specify all keyword arguments or none')

        self.is_input = all(kwarg is None for kwarg in kwargs)

        self.X = X
        self.k = k
        self.m = m
        self.epsilon = l1_eps
        self.W = W

        if self.is_input: 
            self.nu = W(Variable(torch.zeros(self.k*self.m, W.in_features).cauchy_()).type_as(self.X)).unsqueeze(0)
            # self.nu = W(Variable(X.data.new(k*m, W.in_features).cauchy_())).unsqueeze(0)
            self.nu_one = None
        else:
            self.nu = Variable(torch.zeros(1, self.k*self.m, d.size(1)).cauchy_()).type_as(self.X)
            # self.nu = Variable(X.data.new(1, k*m, d.size(1)).cauchy_())
            self.nu_one = Variable(X.data.new(1, d.size(1)).fill_(1))

            if  (~I.data).sum() > 0: 
                self.nu[:,:,(~I.data).nonzero().squeeze(1)] = 0
                self.nu_one[:, (~I.data).nonzero().squeeze(1)] = 0
            self.nu = zl.unsqueeze(1)*self.nu
            self.nu_one = zl*self.nu_one

            self.nu = batch(W(unbatch(self.nu)), self.X.size(0))
            self.nu_one = W(self.nu_one)

    def apply(self, W, d):
        n = self.X.size(0)
        if self.is_input: 
            self.nu = batch(W(unbatch(d.unsqueeze(1)*self.nu)), n)
        else:
            self.nu = batch(W(unbatch(d.unsqueeze(1) * self.nu)),n)
            self.nu_one = W(d * self.nu_one)


    def nu_zlu(self): 
        n = self.l1_norm()
        no = self.nu_one

        nu_zl = (-n + no)/2 
        nu_zu = (-n - no)/2 

        return nu_zl,nu_zu


class L1_median(L1_Cauchy): 
    def l1_norm(self): 
        # return torch.median(nu_hat_1.abs(), 1)[0]/(1-self.epsilon)
        nu_hat_1 = self.nu.view(-1, self.m, self.k, self.nu.size(2))
        return torch.max(torch.median(nu_hat_1.abs(), 2)[0], 1)[0]/(1-self.epsilon)

class L1_geometric(L1_Cauchy): 
    def l1_norm(self): 
        batch_size = self.X.size(0)
        nu = self.nu.view(-1, self.m, self.k, self.nu.size(2))
        return (torch.max(torch.exp((torch.log(nu.abs())/self.k).sum(2))/(1-self.epsilon), 1)[0])

# class L1_Cauchy(Norm):
#     def __init__(self, X, k, m, l1_eps): 
#         self.X = X
#         self.k = k
#         self.m = m
#         self.epsilon = l1_eps
#         self.nu = []
#         self.nu_one = []

#     def input(self, in_features): 
#         return Variable(torch.zeros(self.k*self.m, in_features).cauchy_()).type_as(self.X)

#     def add_layer(self, W, I, d, scatter_grad, zl=[]):
#         B = Variable(torch.zeros(1, self.k*self.m, d.size(1)).cauchy_()).type_as(self.X)
#         ones = Variable(torch.ones(1, d.size(1))).type_as(self.X)

#         if  (~I[-1].data).sum() > 0: 
#             B[:,:,(~I[-1].data).nonzero().squeeze(1)] = 0
#             ones[:, (~I[-1].data).nonzero().squeeze(1)] = 0
#         B = zl[-1].unsqueeze(1)*B
#         ones = zl[-1]*ones

#         self.nu.append(W(unbatch(B)))
#         self.nu_one.append(W(ones))

#     def skip(self):
#         self.nu.append(None)
#         self.nu_one.append(None)

#     def apply(self, W, d):
#         for j in range(len(self.nu)): 
#             if self.nu[j] is not None: 
#                 self.nu[j] = W(unbatch(d.unsqueeze(1) * batch(self.nu[j],
#                    self.X.size(0))))
#                 self.nu_one[j] = W(d * self.nu_one[j])

#     def nu_zlu(self, zl): 
#         terms = [(self.l1_norm(batch(n, self.X.size(0))),no) 
#                      for n,no in zip(self.nu, self.nu_one) if n is not None]
#         nu_zl = sum((-n + no)/2 for n,no in terms)
#         nu_zu = sum((-n - no)/2 for n,no in terms)
#         return nu_zl,nu_zu


#     def nu_zl(self, zl):
#         return sum([(-self.l1_norm(batch(n, self.X.size(0))) + no)/2 
#                      for n,no in zip(self.nu, self.nu_one) if n is not None])

#     def nu_zu(self, zl):
#         return sum([(-self.l1_norm(batch(n, self.X.size(0))) - no)/2 
#                      for n,no in zip(self.nu, self.nu_one) if n is not None])

# class L1_median(L1_Cauchy): 
#     def l1_norm(self): 
#         # return torch.median(nu_hat_1.abs(), 1)[0]/(1-self.epsilon)
#         nu = nu_hat_1.view(-1, self.m, self.k, nu_hat_1.size(2))
#         return torch.max(torch.median(nu.abs(), 2)[0], 1)[0]/(1-self.epsilon)

# class L1_geometric(L1_Cauchy): 
#     def l1_norm(self): 
#         batch_size = self.X.size(0)
#         nu = self.nu.view(-1, self.m, self.k, self.nu.size(2))
#         # original = (torch.exp((torch.log(nu_hat_1[0:1].abs())/k).sum(1)))/(1-self.epsilon)
#         new = (torch.max(torch.exp((torch.log(nu.abs())/self.k).sum(2))/(1-self.epsilon), 1)[0])
#         return new


# class L1_geometric(L1_Cauchy): 
#     def add_layer(self, W, I, d, scatter_grad, zl=[]):
#         return super(L1_geometric, self).add_layer(W, I, d, scatter_grad, zl=zl)

#     def l1_norm(self, nu_hat_1): 
#         batch_size = self.X.size(0)
#         nu_hat_1 = nu_hat_1.view(-1, self.m, self.k, nu_hat_1.size(2))
#         # original = (torch.exp((torch.log(nu_hat_1[0:1].abs())/k).sum(1)))/(1-self.epsilon)
#         new = (torch.max(torch.exp((torch.log(nu_hat_1.abs())/self.k).sum(2))/
#             (1-self.epsilon), 1)[0])
#         return new


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