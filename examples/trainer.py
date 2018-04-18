import torch
import torch.nn as nn
from torch.autograd import Variable
from convex_adversarial import robust_loss

import numpy as np

def train_robust(loader, model, opt, epsilon, epoch, log, verbose, 
                 **kwargs):
    model.train()
    if epoch == 0:
        blank_state = opt.state_dict()



    for i, (X,y) in enumerate(loader):
        X,y = X.cuda(), y.cuda().long()
        if y.dim() == 2: 
            y = y.squeeze(1)
        import time
        start_time = time.time()
        robust_ce, robust_err = robust_loss(model, epsilon, 
                                             Variable(X), Variable(y), 
                                             **kwargs)
        print(time.time()-start_time)
        assert False

        out = model(Variable(X))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.data.max(1)[1] != y).float().sum()  / X.size(0)

        opt.zero_grad()
        robust_ce.backward()

        opt.step()

        print(epoch, i, robust_ce.data[0], robust_err, ce.data[0], err, file=log)

        if verbose and i % verbose == 0: 
            print(epoch, i, robust_ce.data[0], robust_err, ce.data[0], err)
        log.flush()

        del X, y, robust_ce, out, ce
    torch.cuda.empty_cache()


def evaluate_robust(loader, model, epsilon, epoch, log, verbose, **kwargs):
    model.eval()
    all_robust_err = []
    for i, (X,y) in enumerate(loader):
        X,y = X.cuda(), y.cuda().long()
        if y.dim() == 2: 
            y = y.squeeze(1)
        robust_ce, robust_err = robust_loss(model, epsilon, 
                                            Variable(X, volatile=True), 
                                            Variable(y, volatile=True),
                                             **kwargs)
        out = model(Variable(X))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.data.max(1)[1] != y).float().sum()  / X.size(0)

        print(epoch, i, robust_ce.data[0], robust_err, ce.data[0], err, file=log)
        if verbose and i % verbose == 0: 
            print(epoch, i, robust_ce.data[0], robust_err, ce.data[0], err)
        log.flush()

        all_robust_err.append(robust_err)
        del X, y, robust_ce, out, ce
    torch.cuda.empty_cache()
    return (sum(all_robust_err)/len(all_robust_err))

def train_baseline(loader, model, opt, epoch, log, verbose):
    model.train()
    if epoch == 0:
        blank_state = opt.state_dict()

    for i, (X,y) in enumerate(loader):
        X,y = X.cuda(), y.cuda()
        out = model(Variable(X))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.data.max(1)[1] != y).float().sum()  / X.size(0)

        opt.zero_grad()
        ce.backward()
        opt.step()

        print(epoch, i, ce.data[0], err, file=log)
        if verbose and i % verbose == 0: 
            print(epoch, i, ce.data[0], err)
        log.flush()

def evaluate_baseline(loader, model, epoch, log, verbose):
    model.eval()
    all_err = []
    for i, (X,y) in enumerate(loader):
        X,y = X.cuda(), y.cuda()
        out = model(Variable(X))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.data.max(1)[1] != y).float().sum()  / X.size(0)

        all_err.append(err)
        print(epoch, i, ce.data[0], err, file=log)
        if verbose and i % verbose == 0: 
            print(epoch, i, ce.data[0], err)
        log.flush()
    return (sum(all_err)/len(all_err))
