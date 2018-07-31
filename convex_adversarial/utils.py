import torch.nn as nn

###########################################
# Helper function to extract fully        #
# shaped bias terms                       #
###########################################

def full_bias(l, n=None): 
    # expands the bias to the proper size. For convolutional layers, a full
    # output dimension of n must be specified. 
    if isinstance(l, nn.Linear): 
        return l.bias.view(1,-1)
    elif isinstance(l, nn.Conv2d): 
        if n is None: 
            raise ValueError("Need to pass n=<output dimension>")
        b = l.bias.unsqueeze(1).unsqueeze(2)
        if isinstance(n, int): 
            k = int((n/(b.numel()))**0.5)
            return b.expand(1,b.numel(),k,k).contiguous().view(1,-1)
        else: 
            return b.expand(1,*n)
    elif isinstance(l, Dense): 
        return sum(full_bias(layer, n=n) for layer in l.Ws if layer is not None)
    elif isinstance(l, nn.Sequential) and len(l) == 0: 
        return 0
    else:
        raise ValueError("Full bias can't be formed for given layer.")

###########################################
# Sequential models with skip connections #
###########################################

class DenseSequential(nn.Sequential): 
    def forward(self, x):
        xs = [x]
        for module in self._modules.values():
            if 'Dense' in type(module).__name__:
                xs.append(module(*xs))
            else:
                xs.append(module(xs[-1]))
        return xs[-1]

class Dense(nn.Module): 
    def __init__(self, *Ws): 
        super(Dense, self).__init__()
        self.Ws = nn.ModuleList(list(Ws))
        if len(Ws) > 0 and hasattr(Ws[0], 'out_features'): 
            self.out_features = Ws[0].out_features

    def forward(self, *xs): 
        xs = xs[-len(self.Ws):]
        out = sum(W(x) for x,W in zip(xs, self.Ws) if W is not None)
        return out

#######################################
# Epsilon for high probability bounds #
#######################################
import numpy as np
import time

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

def epsilon_from_model(model, X, k, delta, m): 
    if k is None or m is None: 
        raise ValueError("k and m must not be None. ")
    if delta is None: 
        print('No delta specified, not using probabilistic bounds.')
        return 0
        
    X = X[0].unsqueeze(0)
    out_features = []
    for l in model: 
        X = l(X)
        if isinstance(l, (nn.Linear, nn.Conv2d)): 
            out_features.append(X.numel())

    num_est = sum(n for n in out_features[:-1] if k*m < n)

    num_est += sum(n*i for i,n in enumerate(out_features[:-1]) if k*m < n)
    print(num_est)

    sub_delta = (delta/num_est)**(1/m)
    l1_eps = get_epsilon(sub_delta, k)

    if num_est == 0: 
        return 0
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