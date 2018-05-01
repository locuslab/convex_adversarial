import torch
import torch.nn as nn
from torch.autograd import Variable
from convex_adversarial import robust_loss
import torch.optim as optim

import numpy as np
import time
import gc

from attacks import _pgd

def train_robust(loader, model, opt, epsilon, epoch, log, verbose, 
                 **kwargs):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()
    robust_losses = AverageMeter()
    robust_errors = AverageMeter()

    model.train()

    end = time.time()
    for i, (X,y) in enumerate(loader):
        X,y = X.cuda(), y.cuda().long()
        if y.dim() == 2: 
            y = y.squeeze(1)
        data_time.update(time.time() - end)

        out = model(Variable(X, volatile=True))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.data.max(1)[1] != y).float().sum()  / X.size(0)


        robust_ce, robust_err = robust_loss(model, epsilon, 
                                             Variable(X), Variable(y), 
                                             **kwargs)
        opt.zero_grad()
        robust_ce.backward()
        opt.step()

        # measure accuracy and record loss
        losses.update(ce.data[0], X.size(0))
        errors.update(err, X.size(0))
        robust_losses.update(robust_ce.data[0], X.size(0))
        robust_errors.update(robust_err, X.size(0))

        # measure elapsed time
        batch_time.update(time.time()-end)
        end = time.time()

        print(epoch, i, robust_ce.data[0], robust_err, ce.data[0], err, file=log)

        if verbose and i % verbose == 0: 
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Robust loss {rloss.val:.4f} ({rloss.avg:.4f})\t'
                  'Robust error {rerrors.val:.3f} ({rerrors.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error {errors.val:.3f} ({errors.avg:.3f})'.format(
                   epoch, i, len(loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, errors=errors, 
                   rloss = robust_losses, rerrors = robust_errors))
        log.flush()

        del X, y, robust_ce, out, ce, err, robust_err
    torch.cuda.empty_cache()


def evaluate_robust(loader, model, epsilon, epoch, log, verbose, **kwargs):
    batch_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()
    robust_losses = AverageMeter()
    robust_errors = AverageMeter()

    model.eval()

    end = time.time()
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

        _,pgd_err = _pgd(model, Variable(X), Variable(y), epsilon)

        # measure accuracy and record loss
        losses.update(ce.data[0], X.size(0))
        errors.update(err, X.size(0))
        robust_losses.update(robust_ce.data[0], X.size(0))
        robust_errors.update(robust_err, X.size(0))

        # measure elapsed time
        batch_time.update(time.time()-end)
        end = time.time()

        print(epoch, i, robust_ce.data[0], robust_err, ce.data[0], err,
           file=log)
        if verbose and i % verbose == 0: 
            # print(epoch, i, robust_ce.data[0], robust_err, ce.data[0], err)
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Robust loss {rloss.val:.3f} ({rloss.avg:.3f})\t'
                  'Robust error {rerrors.val:.3f} ({rerrors.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error {error.val:.3f} ({error.avg:.3f})'.format(
                      i, len(loader), batch_time=batch_time, 
                      loss=losses, error=errors, rloss = robust_losses, 
                      rerrors = robust_errors))
        log.flush()

        del X, y, robust_ce, out, ce

    print(' * Robust error {rerror.avg:.3f}\t'
          'Error {error.avg:.3f}'
          .format(rerror=robust_errors, error=errors))
    torch.cuda.empty_cache()

def train_baseline(loader, model, opt, epoch, log, verbose):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()

    model.train()

    end = time.time()
    for i, (X,y) in enumerate(loader):
        X,y = X.cuda(), y.cuda()
        data_time.update(time.time() - end)

        out = model(Variable(X))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.data.max(1)[1] != y).float().sum()  / X.size(0)

        opt.zero_grad()
        ce.backward()
        opt.step()

        batch_time.update(time.time()-end)
        end = time.time()
        losses.update(ce.data, X.size(0))
        errors.update(err, X.size(0))

        print(epoch, i, ce.data[0], err, file=log)
        if verbose and i % verbose == 0: 
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error {errors.val:.3f} ({errors.avg:.3f})'.format(
                   epoch, i, len(loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, errors=errors))
        log.flush()

def evaluate_baseline(loader, model, epoch, log, verbose):
    batch_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()

    model.eval()

    end = time.time()
    for i, (X,y) in enumerate(loader):
        X,y = X.cuda(), y.cuda()
        out = model(Variable(X))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.data.max(1)[1] != y).float().sum()  / X.size(0)

        # print to logfile
        print(epoch, i, ce.data[0], err, file=log)

        # measure accuracy and record loss
        losses.update(ce.data[0], X.size(0))
        errors.update(err, X.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if verbose and i % verbose == 0: 
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error {error.val:.3f} ({error.avg:.3f})'.format(
                      i, len(loader), batch_time=batch_time, loss=losses,
                      error=errors))
        log.flush()

    print(' * Error {error.avg:.3f}'
          .format(error=errors))



def train_madry(loader, model, epsilon, opt, epoch, log, verbose):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()
    plosses = AverageMeter()
    perrors = AverageMeter()

    model.train()

    end = time.time()
    for i, (X,y) in enumerate(loader):
        X,y = X.cuda(), y.cuda()
        data_time.update(time.time() - end)

        # # perturb 
        X_pgd = Variable(X, requires_grad=True)
        for _ in range(50): 
            opt_pgd = optim.Adam([X_pgd], lr=1e-3)
            opt.zero_grad()
            loss = nn.CrossEntropyLoss()(model(X_pgd), Variable(y))
            loss.backward()
            eta = 0.01*X_pgd.grad.data.sign()
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            
            # adjust to be within [-epsilon, epsilon]
            eta = torch.clamp(X_pgd.data - X, -epsilon, epsilon)
            X_pgd.data = X + eta
            X_pgd.data = torch.clamp(X_pgd.data, 0, 1)

        out = model(Variable(X))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.data.max(1)[1] != y).float().sum()  / X.size(0)

        pout = model(Variable(X_pgd.data))
        pce = nn.CrossEntropyLoss()(pout, Variable(y))
        perr = (pout.data.max(1)[1] != y).float().sum()  / X.size(0)

        opt.zero_grad()
        pce.backward()
        opt.step()

        batch_time.update(time.time()-end)
        end = time.time()
        losses.update(ce.data[0], X.size(0))
        errors.update(err, X.size(0))
        plosses.update(pce.data[0], X.size(0))
        perrors.update(perr, X.size(0))

        print(epoch, i, ce.data[0], err, file=log)
        if verbose and i % verbose == 0: 
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'PGD Loss {ploss.val:.4f} ({ploss.avg:.4f})\t'
                  'PGD Error {perrors.val:.3f} ({perrors.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error {errors.val:.3f} ({errors.avg:.3f})'.format(
                   epoch, i, len(loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, errors=errors, 
                   ploss=plosses, perrors=perrors))
        log.flush()

def evaluate_madry(loader, model, epsilon, epoch, log, verbose):
    batch_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()
    perrors = AverageMeter()

    model.eval()

    end = time.time()
    for i, (X,y) in enumerate(loader):
        X,y = X.cuda(), y.cuda()
        out = model(Variable(X))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.data.max(1)[1] != y).float().sum()  / X.size(0)


        # # perturb 
        _, pgd_err = _pgd(model, Variable(X), Variable(y), epsilon)

        # print to logfile
        print(epoch, i, ce.data[0], err, file=log)

        # measure accuracy and record loss
        losses.update(ce.data[0], X.size(0))
        errors.update(err, X.size(0))
        perrors.update(pgd_err, X.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if verbose and i % verbose == 0: 
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'PGD Error {perror.val:.3f} ({perror.avg:.3f})\t'
                  'Error {error.val:.3f} ({error.avg:.3f})'.format(
                      i, len(loader), batch_time=batch_time, loss=losses,
                      error=errors, perror=perrors))
        log.flush()

    print(' * PGD error {perror.avg:.3f}\t'
          'Error {error.avg:.3f}'
          .format(error=errors, perror=perrors))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

