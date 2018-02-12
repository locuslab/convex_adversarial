import torch
import torch.nn as nn
from torch.autograd import Variable
from convex_adversarial import robust_loss

import numpy as np

def train_robust(loader, model, opt, epsilon, epoch, log, verbose, 
                 alpha_grad, scatter_grad, l1_proj):
    model.train()
    if epoch == 0:
        blank_state = opt.state_dict()

    for i, (X,y) in enumerate(loader):
        X,y = X.cuda(), y.cuda().long()
        if y.dim() == 2: 
            y = y.squeeze(1)

        robust_ce, robust_err = robust_loss(model, epsilon, 
                                             Variable(X), Variable(y), 
                                             alpha_grad=alpha_grad, 
                                             scatter_grad=scatter_grad,
                                             l1_proj=l1_proj)

        out = model(Variable(X))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.data.max(1)[1] != y).float().sum()  / X.size(0)

        opt.zero_grad()
        robust_ce.backward()

        opt.step()

        print(epoch, i, robust_ce.data[0], robust_err, ce.data[0], err, file=log)

        if i % verbose == 0: 
            print(epoch, i, robust_ce.data[0], robust_err, ce.data[0], err)
        log.flush()

        del X, y, robust_ce, out, ce


def evaluate_robust(loader, model, epsilon, epoch, log, verbose):
    model.eval()
    for i, (X,y) in enumerate(loader):
        X,y = X.cuda(), y.cuda().long()
        if y.dim() == 2: 
            y = y.squeeze(1)
        robust_ce, robust_err = robust_loss(model, epsilon, 
                                            Variable(X, volatile=True), 
                                            Variable(y, volatile=True),
                                             alpha_grad=True, 
                                             scatter_grad=True,
                                             l1_proj=None)
        out = model(Variable(X))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.data.max(1)[1] != y).float().sum()  / X.size(0)

        print(epoch, i, robust_ce.data[0], robust_err, ce.data[0], err, file=log)
        if i % verbose == 0: 
            print(epoch, i, robust_ce.data[0], robust_err, ce.data[0], err)
        log.flush()

        del X, y, robust_ce, out, ce

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
        if i % verbose == 0: 
            print(epoch, i, ce.data[0], err)
        log.flush()

def evaluate_baseline(loader, model, epoch, log, verbose):
    model.eval()
    for i, (X,y) in enumerate(loader):
        X,y = X.cuda(), y.cuda()
        out = model(Variable(X))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.data.max(1)[1] != y).float().sum()  / X.size(0)

        print(epoch, i, ce.data[0], err, file=log)
        if i % verbose == 0: 
            print(epoch, i, ce.data[0], err)
        log.flush()
