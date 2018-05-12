import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import setproctitle
import argparse

import problems as pblm
from trainer import *

if __name__ == "__main__": 
    args = pblm.argparser(opt='adam', verbose=200, epsilon=0.01, starting_epsilon=0.001)
    setproctitle.setproctitle(args.prefix) 

    train_log = open(args.prefix + "_train.log", "w")
    test_log = open(args.prefix + "_test.log", "w")

    train_loader, test_loader = pblm.svhn_loaders(args.batch_size)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # new svhn
    if args.model == 'deep': 
        model = pblm.svhn_model_deep(args.model_factor).cuda()
    elif args.model == 'wide': 
        model = pblm.svhn_model_wide(args.model_factor).cuda()
    else: 
        model = pblm.svhn_model().cuda()

    starting_epoch=0
    if args.resume: 
        checkpoint = torch.load(args.prefix + '_checkpoint.pth')
        model.load_state_dict(checkpoint['state_dict'])
        starting_epoch = checkpoint['epoch']+1

    kwargs = pblm.args2kwargs(args)
    best_err = 1

    if args.eval is not None: 
        try: 
            model.load_state_dict(torch.load(args.eval))
        except:
            print('[Warning] eval argument could not be loaded, evaluating a random model')
        evaluate_robust(test_loader, model, args.epsilon, 0, test_log,
                        args.verbose, **kwargs)
    else: 
        opt = optim.Adam(model.parameters(), lr=args.lr)
        # opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
            # weight_decay=args.weight_decay)
        lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)
        eps_schedule = np.logspace(np.log10(args.starting_epsilon), 
                                   np.log10(args.epsilon), 
                                   args.schedule_length)
        for t in range(starting_epoch, args.epochs):
            lr_scheduler.step(epoch=max(t-len(eps_schedule), 0))
            if args.method == 'baseline': 
                train_baseline(train_loader, model, opt, t, train_log, args.verbose)
                err = evaluate_baseline(test_loader, model, t, test_log, args.verbose)
            else:
                if t < len(eps_schedule) and args.starting_epsilon is not None: 
                    # epsilon = args.starting_epsilon + (t/(args.epochs//2))*(args.epsilon - args.starting_epsilon)
                    epsilon = float(eps_schedule[t])
                else:
                    epsilon = args.epsilon
                train_robust(train_loader, model, opt, epsilon, t, train_log, 
                    args.verbose, l1_type=args.l1_train, **kwargs)
                err = evaluate_robust(test_loader, model, args.epsilon, t, test_log,
                   args.verbose, l1_type=args.l1_test, **kwargs)
            print('Epoch {}: {} err'.format(t, err))
            
            if err < best_err: 
                best_err = err
                torch.save({
                    'state_dict' : model.state_dict, 
                    'err' : best_err,
                    'epoch' : t
                    }, args.prefix + "_best.pth")
                
            torch.save({ 
                'state_dict': model.state_dict(),
                'err' : err,
                'epoch' : t
                }, args.prefix + "_checkpoint.pth")