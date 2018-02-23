import torch
from torch.autograd import Variable
from abc import ABC, abstractmethod

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

    def nu_zl(self, zl, I):
        return sum([(zl[j][I[j]] * (-self.nu[j].t()).clamp(min=0)).mm(self.I_collapse[j]).t()
                                for j in range(len(zl)) if not self.nu[j] is None])

    def nu_zu(self, zl, I):
        return sum([(zl[j][I[j]] * self.nu[j].t().clamp(min=0)).mm(self.I_collapse[j]).t()
                                for j in range(len(zl)) if not self.nu[j] is None])

class L1_Cauchy(Norm):
    def __init__(self, X): 
        self.X = X
        self.nu = []
        self.nu_one = []

    def input(self, in_features, k=100): 
        return Variable(torch.zeros(k, in_features).cauchy_()).type_as(self.X)

    def add_layer(self, W, I, d, scatter_grad, zl=[], k=100):
        B = Variable(torch.zeros(1, k, d.size(1)).cauchy_()).type_as(self.X)
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


    def nu_zl(self, zl):
        return sum([(-self.l1_norm(batch(n, self.X.size(0))) + no)/2 
                     for n,no in zip(self.nu, self.nu_one) if n is not None])

    def nu_zu(self, zl):
        return sum([(-self.l1_norm(batch(n, self.X.size(0))) - no)/2 
                     for n,no in zip(self.nu, self.nu_one) if n is not None])

class L1_median(L1_Cauchy): 
    def l1_norm(self, nu_hat_1): 
        return torch.median(nu_hat_1.abs(), 1)[0]

class L1_geometric(L1_Cauchy): 
    def l1_norm(self, nu_hat_1): 
        k = nu_hat_1.size(1)
        return torch.exp((torch.log(nu_hat_1.abs())/k).sum(1))