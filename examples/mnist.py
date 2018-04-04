import waitGPU
# import setGPU
# waitGPU.wait(utilization=20, available_memory=10000, interval=60)
waitGPU.wait(gpu_ids=[3])

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
from convex_adversarial import epsilon_from_model

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--starting_epsilon", type=float, default=0.05)
    parser.add_argument('--prefix')
    parser.add_argument('--eval')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--alpha_grad', action='store_true')
    parser.add_argument('--scatter_grad', action='store_true')
    parser.add_argument('--l1_proj', type=int, default=None)
    parser.add_argument('--delta', type=float, default=None)
    parser.add_argument('--m', type=int, default=None)
    parser.add_argument('--large', action='store_true')
    parser.add_argument('--vgg', action='store_true')
    args = parser.parse_args()

    if args.prefix: 
        if args.vgg: 
            args.prefix += '_vgg'
        elif args.large: 
            args.prefix += '_large'

        banned = ['alpha_grad', 'scatter_grad', 'verbose', 'prefix',
                  'large', 'vgg']
        for arg in sorted(vars(args)): 
            if arg not in banned: 
                args.prefix += '_' + arg + '_' +str(getattr(args, arg))
    else: 
        args.prefix = 'mnist_temporary'
    print("saving file to {}".format(args.prefix))
    # args.prefix = args.prefix or 'mnist_conv_{:.4f}_{:.4f}_0'.format(args.epsilon, args.lr).replace(".","_")
    setproctitle.setproctitle(args.prefix)

    train_log = open(args.prefix + "_train.log", "w")
    test_log = open(args.prefix + "_test.log", "w")

    train_loader, test_loader = pblm.mnist_loaders(args.batch_size)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if args.vgg: 
        model = pblm.mnist_model_vgg().cuda()
    elif args.large: 
        model = pblm.mnist_model_large().cuda()
    else: 
        model = pblm.mnist_model().cuda() 

    if args.l1_proj is not None: 
        for X,y in train_loader: 
            break
        l1_eps = epsilon_from_model(model, Variable(X.cuda()), args.l1_proj,
                                    args.delta, args.m)
        print('''
With probability {} and projection into {} dimensions and a max
over {} estimates, we have epsilon={}'''.format(args.delta, args.l1_proj,
                                                args.m, l1_eps))
        kwargs = {
            'alpha_grad' : args.alpha_grad,
            'scatter_grad' : args.scatter_grad, 
            'l1_proj' : args.l1_proj, 
            'l1_eps' : l1_eps, 
            'm' : args.m
        }
    else:
        kwargs = {
            'alpha_grad' : args.alpha_grad,
            'scatter_grad' : args.scatter_grad, 
        }

    if args.eval is not None: 
        try: 
            model.load_state_dict(torch.load(args.eval))
        except:
            print('[Warning] eval argument could not be loaded, evaluating a random model')
        evaluate_robust(test_loader, model, args.epsilon, 0, test_log,
            args.verbose, 
              **kwargs)
    else: 
        opt = optim.Adam(model.parameters(), lr=args.lr)
        for t in range(args.epochs):
            if args.baseline: 
                train_baseline(train_loader, model, opt, t, train_log, args.verbose)
                evaluate_baseline(test_loader, model, t, test_log, args.verbose)
            else:
                if t <= args.epochs//2 and args.starting_epsilon is not None: 
                    epsilon = args.starting_epsilon + (t/(args.epochs//2))*(args.epsilon - args.starting_epsilon)
                else:
                    epsilon = args.epsilon
                train_robust(train_loader, model, opt, epsilon, t, train_log, 
                    args.verbose, **kwargs)
                evaluate_robust(test_loader, model, args.epsilon, t, test_log,
                   args.verbose, **kwargs)
                      # l1_geometric=args.l1_proj)

            torch.save(model.state_dict(), args.prefix + "_model.pth")