import waitGPU
waitGPU.wait(utilization=20, available_memory=10000, interval=10)#, gpu_ids=[2,3])

# import setGPU
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--starting_epsilon", type=float, default=0.05)
    parser.add_argument('--prefix')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--alpha_grad', action='store_true')
    parser.add_argument('--scatter_grad', action='store_true')
    parser.add_argument('--l1_median', type=int, default=None)
    parser.add_argument('--l1_geometric', type=int, default=None)
    parser.add_argument('--l1', type=int, default=None)
    parser.add_argument('--large', action='store_true')
    parser.add_argument('--vgg', action='store_true')
    args = parser.parse_args()
    args.prefix = args.prefix or 'mnist_conv_{:.4f}_{:.4f}_0'.format(args.epsilon, args.lr).replace(".","_")
    setproctitle.setproctitle(args.prefix)

    train_log = open(args.prefix + "_train.log", "w")
    test_log = open(args.prefix + "_test.log", "w")

    train_loader, _ = pblm.mnist_loaders(args.batch_size)
    _, test_loader = pblm.mnist_loaders(args.batch_size)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.large: 
        model = pblm.mnist_model_large().cuda()
        # for X,y in train_loader: 
        #     print(model(Variable(X.cuda())).size())
        #     assert False

    elif args.vgg: 
        model = pblm.mnist_model_vgg().cuda()
    else: 
        model = pblm.mnist_model().cuda()

    kwargs = {
        'alpha_grad' : args.alpha_grad, 
        'scatter_grad' : args.scatter_grad, 
        # 'l1' : args.l1
        'l1_median' : args.l1_median, 
        'l1_geometric' : args.l1_geometric
    }

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
                args.verbose,  **kwargs)
            evaluate_robust(test_loader, model, args.epsilon, t, test_log,
                            args.verbose, **kwargs)

        torch.save(model.state_dict(), args.prefix + "_model.pth")