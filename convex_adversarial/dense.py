import torch.nn as nn

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
