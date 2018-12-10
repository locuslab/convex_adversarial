
import torch
import torch.nn as nn
from torch.autograd import Variable
from convex_adversarial import robust_loss, robust_loss_parallel
import torch.optim as optim

import numpy as np
import time
import gc

from attacks import _pgd

DEBUG = False

def train_robust(loader, model, opt, epsilon, epoch, log, verbose, 
                real_time=False, clip_grad=None, **kwargs):
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

        with torch.no_grad(): 
            out = model(Variable(X))
            ce = nn.CrossEntropyLoss()(out, Variable(y))
            err = (out.max(1)[1] != y).float().sum()  / X.size(0)


        robust_ce, robust_err = robust_loss(model, epsilon, 
                                             Variable(X), Variable(y), 
                                             **kwargs)
        opt.zero_grad()
        robust_ce.backward()


        if clip_grad: 
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        opt.step()

        # measure accuracy and record loss
        losses.update(ce.item(), X.size(0))
        errors.update(err.item(), X.size(0))
        robust_losses.update(robust_ce.detach().item(), X.size(0))
        robust_errors.update(robust_err, X.size(0))

        # measure elapsed time
        batch_time.update(time.time()-end)
        end = time.time()

        print(epoch, i, robust_ce.detach().item(), 
                robust_err, ce.item(), err.item(), file=log)

        if verbose and (i % verbose == 0 or real_time): 
            endline = '\n' if i % verbose == 0 else '\r'
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Robust loss {rloss.val:.4f} ({rloss.avg:.4f})\t'
                  'Robust error {rerrors.val:.3f} ({rerrors.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error {errors.val:.3f} ({errors.avg:.3f})'.format(
                   epoch, i, len(loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, errors=errors, 
                   rloss = robust_losses, rerrors = robust_errors), end=endline)
        log.flush()

        del X, y, robust_ce, out, ce, err, robust_err
        if DEBUG and i ==10: 
            break
    print('')
    torch.cuda.empty_cache()


def evaluate_robust(loader, model, epsilon, epoch, log, verbose, 
                    real_time=False, parallel=False, **kwargs):
    batch_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()
    robust_losses = AverageMeter()
    robust_errors = AverageMeter()

    model.eval()

    end = time.time()

    torch.set_grad_enabled(False)
    for i, (X,y) in enumerate(loader):
        X,y = X.cuda(), y.cuda().long()
        if y.dim() == 2: 
            y = y.squeeze(1)

        robust_ce, robust_err = robust_loss(model, epsilon, X, y, **kwargs)

        out = model(Variable(X))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.max(1)[1] != y).float().sum()  / X.size(0)

        # _,pgd_err = _pgd(model, Variable(X), Variable(y), epsilon)

        # measure accuracy and record loss
        losses.update(ce.item(), X.size(0))
        errors.update(err, X.size(0))
        robust_losses.update(robust_ce.item(), X.size(0))
        robust_errors.update(robust_err, X.size(0))

        # measure elapsed time
        batch_time.update(time.time()-end)
        end = time.time()

        print(epoch, i, robust_ce.item(), robust_err, ce.item(), err.item(),
           file=log)
        if verbose: 
            # print(epoch, i, robust_ce.data[0], robust_err, ce.data[0], err)
            endline = '\n' if i % verbose == 0 else '\r'
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Robust loss {rloss.val:.3f} ({rloss.avg:.3f})\t'
                  'Robust error {rerrors.val:.3f} ({rerrors.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error {error.val:.3f} ({error.avg:.3f})'.format(
                      i, len(loader), batch_time=batch_time, 
                      loss=losses, error=errors, rloss = robust_losses, 
                      rerrors = robust_errors), end=endline)
        log.flush()

        del X, y, robust_ce, out, ce
        if DEBUG and i ==10: 
            break
    torch.set_grad_enabled(True)
    torch.cuda.empty_cache()
    print('')
    print(' * Robust error {rerror.avg:.3f}\t'
          'Error {error.avg:.3f}'
          .format(rerror=robust_errors, error=errors))
    return robust_errors.avg

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
        losses.update(ce.data[0], X.size(0))
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
    return errors.avg



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
        losses.update(ce.item(), X.size(0))
        errors.update(err, X.size(0))
        plosses.update(pce.item(), X.size(0))
        perrors.update(perr, X.size(0))

        print(epoch, i, ce.item(), err, file=log)
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
        print(epoch, i, ce.item(), err, file=log)

        # measure accuracy and record loss
        losses.update(ce.item(), X.size(0))
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
    return errors.avg


def robust_loss_cascade(models, epsilon, X, y, **kwargs): 
    total_robust_ce = 0.
    total_ce = 0.
    total_robust_err = 0.
    total_err = 0.

    batch_size = float(X.size(0))

    I = torch.arange(X.size(0)).type_as(y.data)

    if X.size(0) == 1: 
        rl = robust_loss_parallel
    else:
        rl = robust_loss

    for j,model in enumerate(models[:-1]): 

        out = model(X)
        ce = nn.CrossEntropyLoss(reduce=False)(out, y)

        _, uncertified = rl(model, epsilon, X,
                                     out.max(1)[1],
                                     size_average=False, **kwargs)

        certified = ~uncertified
        l = []
        if certified.sum() == 0: 
            pass
            # print("Warning: Cascade stage {} has no certified values.".format(j+1))
        else: 
            X_cert = X[Variable(certified.nonzero()[:,0])]
            y_cert = y[Variable(certified.nonzero()[:,0])]

            ce = ce[Variable(certified.nonzero()[:,0])]
            out = out[Variable(certified.nonzero()[:,0])]
            err = (out.data.max(1)[1] != y_cert.data).float()
            robust_ce, robust_err = rl(model, epsilon, 
                                                 X_cert, 
                                                 y_cert, 
                                                 size_average=False,
                                                 **kwargs)
            # add statistics for certified examples
            total_robust_ce += robust_ce.sum()
            total_ce += ce.data.sum()
            total_robust_err += robust_err.sum()
            total_err += err.sum()
            l.append(certified.sum())
            # reduce data set to uncertified examples
            if uncertified.sum() > 0: 
                X = X[Variable(uncertified.nonzero()[:,0])]
                y = y[Variable(uncertified.nonzero()[:,0])]
                I = I[uncertified.nonzero()[:,0]]
            else: 
                robust_ce = total_robust_ce/batch_size
                ce = total_ce/batch_size
                robust_err = total_robust_err.item()/batch_size
                err = total_err.item()/batch_size
                return robust_ce, robust_err, ce, err, None
        ####################################################################
    # compute normal ce and robust ce for the last model
    out = models[-1](X)
    ce = nn.CrossEntropyLoss(reduce=False)(out, y)
    err = (out.data.max(1)[1] != y.data).float()

    robust_ce, robust_err = rl(models[-1], epsilon, X, y,
                                         size_average=False, **kwargs)

    # update statistics with the remaining model and take the average 
    total_robust_ce += robust_ce.sum()
    total_ce += ce.data.sum()
    total_robust_err += robust_err.sum()
    total_err += err.sum()

    robust_ce = total_robust_ce/batch_size
    ce = total_ce/batch_size
    robust_err = total_robust_err.item()/batch_size
    err = total_err.item()/batch_size

    _, uncertified = rl(models[-1], epsilon, 
                                 X, 
                                 out.max(1)[1], 
                                 size_average=False,
                                 **kwargs)
    if uncertified.sum() > 0: 
        I = I[uncertified.nonzero()[:,0]]
    else:
        I = None

    return robust_ce, robust_err, ce, err, I

def sampler_robust_cascade(loader, models, epsilon, batch_size, **kwargs): 
    torch.set_grad_enabled(False)
    dataset = loader.dataset 
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    l = []

    start = 0
    total = 0
    for i, (X,y) in enumerate(loader): 
        print('Certifying minibatch {}/{} [current total: {}/{}]'.format(i, len(loader), total, len(dataset)), end='\r')

        X = X.cuda()
        y = y.cuda()

        _, _, _, _, uncertified = robust_loss_cascade(models, epsilon, 
                                                   Variable(X), 
                                                   Variable(y), 
                                                   **kwargs)
        if uncertified is not None: 
            l.append(uncertified+start)
            total += len(uncertified)
        start += X.size(0)
        if DEBUG and i ==10: 
            break
    print('')
    torch.set_grad_enabled(True)
    if len(l) > 0: 
        total = torch.cat(l)
        sampler = torch.utils.data.sampler.SubsetRandomSampler(total)
        return torch.utils.data.DataLoader(dataset, batch_size=loader.batch_size, shuffle=False, pin_memory=True, sampler=sampler)
    else:
        return None

def evaluate_robust_cascade(loader, models, epsilon, epoch, log, verbose, **kwargs):
    batch_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()
    robust_losses = AverageMeter()
    robust_errors = AverageMeter()

    for model in models:
        model.eval()

    torch.set_grad_enabled(False)
    end = time.time()
    for i, (X,y) in enumerate(loader):
        X,y = X.cuda(), y.cuda().long()
        if y.dim() == 2: 
            y = y.squeeze(1)
        robust_ce, robust_err, ce, err, _ = robust_loss_cascade(models, 
                                                             epsilon, 
                                                             Variable(X), 
                                                             Variable(y), 
                                                             **kwargs)

        # measure accuracy and record loss
        losses.update(ce, X.size(0))
        errors.update(err, X.size(0))
        robust_losses.update(robust_ce.item(), X.size(0))
        robust_errors.update(robust_err, X.size(0))

        # measure elapsed time
        batch_time.update(time.time()-end)
        end = time.time()

        print(epoch, i, robust_ce.item(), robust_err, ce.item(), err,
           file=log)
        if verbose: 
            endline = '\n' if  i % verbose == 0 else '\r'
            # print(epoch, i, robust_ce.data[0], robust_err, ce.data[0], err)
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Robust loss {rloss.val:.3f} ({rloss.avg:.3f})\t'
                  'Robust error {rerrors.val:.3f} ({rerrors.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error {error.val:.3f} ({error.avg:.3f})'.format(
                      i, len(loader), batch_time=batch_time, 
                      loss=losses, error=errors, rloss = robust_losses, 
                      rerrors = robust_errors), end=endline)
        log.flush()

        del X, y, robust_ce, ce
        if DEBUG and i == 10: 
            break
    torch.cuda.empty_cache()
    print('')
    print(' * Robust error {rerror.avg:.3f}\t'
          'Error {error.avg:.3f}'
          .format(rerror=robust_errors, error=errors))
    torch.set_grad_enabled(True)
    return robust_errors.avg


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

