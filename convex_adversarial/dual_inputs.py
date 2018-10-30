import torch
import torch.nn as nn

from .dual import DualObject

def select_input(X, epsilon, proj, norm, bounded_input):
    if proj is not None and norm=='l1_median' and X[0].numel() > proj:
        if bounded_input: 
            return InfBallProjBounded(X,epsilon,proj)
        else: 
            return InfBallProj(X,epsilon,proj)
    elif norm == 'l1':
        if bounded_input: 
            return InfBallBounded(X, epsilon)
        else:
            return InfBall(X, epsilon)
    elif proj is not None and norm=='l2_normal' and X[0].numel() > proj: 
        return L2BallProj(X,epsilon,proj)
    elif norm == 'l2': 
        return L2Ball(X,epsilon)
    else: 
        raise ValueError("Unknown estimation type: {}".format(norm))

class InfBall(DualObject):
    def __init__(self, X, epsilon): 
        super(InfBall, self).__init__()
        self.epsilon = epsilon

        n = X[0].numel()
        self.nu_x = [X] 
        self.nu_1 = [X.new(n,n)]
        torch.eye(n, out=self.nu_1[0])
        self.nu_1[0] = self.nu_1[0].view(-1,*X.size()[1:]).unsqueeze(0)

    def apply(self, dual_layer): 
        self.nu_x.append(dual_layer(*self.nu_x))
        self.nu_1.append(dual_layer(*self.nu_1))

    def bounds(self, network=None): 
        if network is None: 
            nu_1 = self.nu_1[-1]
            nu_x = self.nu_x[-1]
        else:
            nu_1 = network(self.nu_1[0])
            nu_x = network(self.nu_x[0])

        epsilon = self.epsilon
        l1 = nu_1.abs().sum(1)
        if isinstance(epsilon, torch.Tensor): 
            while epsilon.dim() < nu_x.dim(): 
                epsilon = epsilon.unsqueeze(1)

        return (nu_x - epsilon*l1, 
                nu_x + epsilon*l1)

    def objective(self, *nus): 
        epsilon = self.epsilon
        nu = nus[-1]
        nu = nu.view(nu.size(0), nu.size(1), -1)
        nu_x = nu.matmul(self.nu_x[0].view(self.nu_x[0].size(0),-1).unsqueeze(2)).squeeze(2)
        if isinstance(self.epsilon, torch.Tensor): 
            while epsilon.dim() < nu.dim()-1: 
                epsilon = epsilon.unsqueeze(1)
        l1 = epsilon*nu.abs().sum(2)
        return -nu_x - l1

class InfBallBounded(DualObject):
    def __init__(self, X, epsilon, l=0, u=1): 
        super(InfBallBounded, self).__init__()
        self.epsilon = epsilon
        self.l = (X-epsilon).clamp(min=l).view(X.size(0), 1, -1)
        self.u = (X+epsilon).clamp(max=u).view(X.size(0), 1, -1)

        n = X[0].numel()
        self.nu_x = [X] 
        self.nu_1 = [X.new(n,n)]
        torch.eye(n, out=self.nu_1[0])
        self.nu_1[0] = self.nu_1[0].view(-1,*X.size()[1:]).unsqueeze(0)

    def apply(self, dual_layer): 
        self.nu_x.append(dual_layer(*self.nu_x))
        self.nu_1.append(dual_layer(*self.nu_1))

    def bounds(self, network=None): 
        if network is None: 
            nu = self.nu_1[-1]
        else:
            nu = network(self.nu_1[0])
        nu_pos = nu.clamp(min=0).view(nu.size(0), nu.size(1), -1)
        nu_neg = nu.clamp(max=0).view(nu.size(0), nu.size(1), -1)

        zu = (self.u.matmul(nu_pos) + self.l.matmul(nu_neg)).squeeze(1)
        zl = (self.u.matmul(nu_neg) + self.l.matmul(nu_pos)).squeeze(1)
        return (zl.view(zl.size(0), *nu.size()[2:]), 
                zu.view(zu.size(0), *nu.size()[2:]))

    def objective(self, *nus): 
        nu = nus[-1]
        nu_pos = nu.clamp(min=0).view(nu.size(0), nu.size(1), -1)
        nu_neg = nu.clamp(max=0).view(nu.size(0), nu.size(1), -1)
        u, l = self.u.unsqueeze(3).squeeze(1), self.l.unsqueeze(3).squeeze(1)
        return (-nu_neg.matmul(l) - nu_pos.matmul(u)).squeeze(2)

class InfBallProj(InfBall):
    def __init__(self, X, epsilon, k): 
        DualObject.__init__(self)
        self.epsilon = epsilon

        n = X[0].numel()
        self.nu_x = [X] 
        self.nu = [X.new(1,k,*X.size()[1:]).cauchy_()]

    def apply(self, dual_layer): 
        self.nu_x.append(dual_layer(*self.nu_x))
        self.nu.append(dual_layer(*self.nu))

    def bounds(self, network=None):
        if network is None: 
            nu = self.nu[-1]
            nu_x = self.nu_x[-1]
        else: 
            nu = network(self.nu[0])
            nu_x = network(self.nu_x[0])

        l1 = torch.median(self.nu[-1].abs(), 1)[0]
        return (nu_x - self.epsilon*l1, 
                nu_x + self.epsilon*l1)

class InfBallProjBounded(InfBallProj):
    def __init__(self, X, epsilon, k, l=0, u=1): 
        self.epsilon = epsilon

        self.nu_one_l = [(X-epsilon).clamp(min=l)]
        self.nu_one_u = [(X+epsilon).clamp(max=u)]
        self.nu_x = [X] 

        self.l = self.nu_one_l[-1].view(X.size(0), 1, -1)
        self.u = self.nu_one_u[-1].view(X.size(0), 1, -1)

        n = X[0].numel()
        R = X.new(1,k,*X.size()[1:]).cauchy_()
        self.nu_l = [R * self.nu_one_l[-1].unsqueeze(1)]
        self.nu_u = [R * self.nu_one_u[-1].unsqueeze(1)]

    def apply(self, dual_layer): 
        self.nu_l.append(dual_layer(*self.nu_l))
        self.nu_one_l.append(dual_layer(*self.nu_one_l))
        self.nu_u.append(dual_layer(*self.nu_u))
        self.nu_one_u.append(dual_layer(*self.nu_one_u))

    def bounds(self, network=None): 
        if network is None: 
            nu_u = self.nu_u[-1]
            nu_one_u = self.nu_one_u[-1]
            nu_l = self.nu_l[-1]
            nu_one_l = self.nu_one_l[-1]
        else: 
            nu_u = network(self.nu_u[0])
            nu_one_u = network(self.nu_one_u[0])
            nu_l = network(self.nu_l[0])
            nu_one_l = network(self.nu_one_l[0])

        nu_l1_u = torch.median(nu_u.abs(),1)[0]
        nu_pos_u = (nu_l1_u + nu_one_u)/2
        nu_neg_u = (-nu_l1_u + nu_one_u)/2

        nu_l1_l = torch.median(nu_l.abs(),1)[0]
        nu_pos_l = (nu_l1_l + nu_one_l)/2
        nu_neg_l = (-nu_l1_l + nu_one_l)/2

        zu = nu_pos_u + nu_neg_l
        zl = nu_neg_u + nu_pos_l
        return zl,zu

# L2 balls
class L2Ball(DualObject):
    def __init__(self, X, epsilon): 
        super(L2Ball, self).__init__()
        self.epsilon = epsilon

        n = X[0].numel()
        self.nu_x = [X] 
        self.nu_1 = [X.new(n,n)]
        torch.eye(n, out=self.nu_1[0])
        self.nu_1[0] = self.nu_1[0].view(-1,*X.size()[1:]).unsqueeze(0)

    def apply(self, dual_layer): 
        self.nu_x.append(dual_layer(*self.nu_x))
        self.nu_1.append(dual_layer(*self.nu_1))

    def bounds(self, network=None): 
        if network is None: 
            nu_1 = self.nu_1[-1]
            nu_x = self.nu_x[-1]
        else:
            nu_1 = network(self.nu_1[0])
            nu_x = network(self.nu_x[0])

        epsilon = self.epsilon
        l2 = nu_1.norm(2, 1)

        if isinstance(epsilon, torch.Tensor): 
            while epsilon.dim() < nu_x.dim(): 
                epsilon = epsilon.unsqueeze(1)

        return (nu_x - epsilon*l2, 
                nu_x + epsilon*l2)

    def objective(self, *nus): 
        epsilon = self.epsilon
        nu = nus[-1]
        nu = nu.view(nu.size(0), nu.size(1), -1)
        nu_x = nu.matmul(self.nu_x[0].view(self.nu_x[0].size(0),-1).unsqueeze(2)).squeeze(2)
        if isinstance(self.epsilon, torch.Tensor): 
            while epsilon.dim() < nu.dim()-1: 
                epsilon = epsilon.unsqueeze(1)

        l2 = nu.norm(2,2)
        return -nu_x - epsilon*l2


class L2BallProj(L2Ball):
    def __init__(self, X, epsilon, k): 
        DualObject.__init__(self)
        self.epsilon = epsilon

        n = X[0].numel()
        self.nu_x = [X] 
        self.nu = [X.new(1,k,*X.size()[1:]).normal_()]

    def apply(self, dual_layer): 
        self.nu_x.append(dual_layer(*self.nu_x))
        self.nu.append(dual_layer(*self.nu))

    def bounds(self, network=None):
        if network is None: 
            nu = self.nu[-1]
            nu_x = self.nu_x[-1]
        else: 
            nu = network(self.nu[0])
            nu_x = network(self.nu_x[0])

        k = nu.size(1)
        l2 = nu.norm(2, 1)/(k**0.5)
        
        return (nu_x - self.epsilon*l2, 
                nu_x + self.epsilon*l2)