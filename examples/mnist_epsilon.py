import waitGPU
waitGPU.wait(utilization=20, available_memory=10000, interval=10)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from convex_adversarial import DualNetBoundsBatch

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import setproctitle
import argparse

import problems as pblm

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--niters', type=int, default=100)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--threshold', type=float, default=1e-5)
    parser.add_argument('--prefix', default='temp')
    parser.add_argument('--train', action='store_true')

    parser.add_argument('--mnist', action='store_true')
    parser.add_argument('--svhn', action='store_true')
    parser.add_argument('--har', action='store_true')
    parser.add_argument('--fashion', action='store_true')

    args = parser.parse_args()

    if args.mnist: 
        train_loader, test_loader = pblm.mnist_loaders(args.batch_size)
        model = pblm.mnist_model().cuda()
        model.load_state_dict(torch.load('icml/mnist_epochs_100_baseline_model.pth'))
        # model.load_state_dict(torch.load('icml/mnist_epochs100_model.pth'))
    elif args.svhn: 
        train_loader, test_loader = pblm.svhn_loaders(args.batch_size)
        model = pblm.svhn_model().cuda()
        model.load_state_dict(torch.load('svhn_new/svhn_epsilon_0_01_schedule_0_001'))
    elif args.har:
        pass
    elif args.fashion: 
        pass
    else:
        raise ValueError("Need to specify which problem.")
    for p in model.parameters(): 
        p.requires_grad = False

    num_classes = model[-1].out_features

    correct = []
    incorrect = []
    l = []

    loader = train_loader if args.train else test_loader

    for j,(X,y) in enumerate(loader): 
        print('*** Batch {} ***'.format(j))
        epsilon = Variable(args.epsilon*torch.ones(args.batch_size).cuda(), requires_grad=True)
        X, y = Variable(X).cuda(), Variable(y).cuda()

        out = Variable(model(X).data.max(1)[1])

        # form c without the 0 row
        c = Variable(torch.eye(num_classes).type_as(X.data)[out.data].unsqueeze(1) - torch.eye(num_classes).type_as(X.data).unsqueeze(0))
        I = (~(out.data.unsqueeze(1) == torch.arange(num_classes).type_as(out.data).unsqueeze(0)).unsqueeze(2))
        c = (c[I].view(args.batch_size,num_classes-1,num_classes))
        if X.is_cuda:
            c = c.cuda()

        alpha = args.alpha

        def f(eps): 
            dual = DualNetBoundsBatch(model, X, eps, True, True)
            f = -dual.g(c)
            return (f.max(1)[0])

        for i in range(args.niters): 
            # dual = DualNetBoundsBatch(model, X, epsilon, True, True)
            
            # f = -dual.g(c)
            # f_max = (f.max(1)[0])

            # f_val = f_max.data.abs().sum()
            f_max = f(epsilon)
            print(i, f_max.data.abs().sum())
            # if done, stop
            if (f_max.data.abs() <= args.threshold).all(): 
                break

            # otherwise, compute gradient and update
            (f_max).sum().backward()

            alpha = args.alpha
            epsilon0 = Variable((epsilon - alpha*(f_max/(epsilon.grad))).data,
               requires_grad=True)

            while (f(epsilon0).data.abs().sum() >= f_max.data.abs().sum()):
                alpha *= 0.5
                epsilon0 = Variable((epsilon - alpha*(f_max/(epsilon.grad))).data,
                                    requires_grad=True)
                if alpha <= 1e-3: 
                    break

            epsilon = epsilon0
            # epsilon = Variable((epsilon - alpha*(f_max/(epsilon.grad))).data,
            #    requires_grad=True)
            # if i % 10 == 9: 
            #     alpha *= 0.9
            del f_max

        if i == args.niters - 1: 
            l.append(j)
        correct.append(epsilon[y==out])
        incorrect.append(epsilon[y!=out])

        del X, y
        # if j > 1: 
        #     break
    print(l)
    torch.save(torch.cat(correct, 0), '{}_correct_epsilons.pth'.format(args.prefix))
    torch.save(torch.cat(incorrect, 0), '{}_incorrect_epsilons.pth'.format(args.prefix))