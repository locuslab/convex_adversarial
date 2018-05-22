import waitGPU
waitGPU.wait(utilization=20, available_memory=10000, interval=10)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from convex_adversarial import DualNetBounds

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import setproctitle
import argparse

import problems as pblm

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--niters', type=int, default=20)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--threshold', type=float, default=1e-4)
    parser.add_argument('--prefix', default='temp')
    parser.add_argument('--train', action='store_true')

    parser.add_argument('--mnist', action='store_true')
    parser.add_argument('--svhn', action='store_true')
    parser.add_argument('--har', action='store_true')
    parser.add_argument('--fashion', action='store_true')
    parser.add_argument('--model')

    args = parser.parse_args()

    if args.mnist: 
        train_loader, test_loader = pblm.mnist_loaders(args.batch_size)
        model = pblm.mnist_model().cuda()
        model.load_state_dict(torch.load('icml/mnist_epochs_100_baseline_model.pth'))
    elif args.svhn: 
        train_loader, test_loader = pblm.svhn_loaders(args.batch_size)
        model = pblm.svhn_model().cuda()
        model.load_state_dict(torch.load('pixel2/svhn_small_batch_size_50_epochs_100_epsilon_0.0078_l1_proj_50_l1_test_median_l1_train_median_lr_0.001_opt_adam_schedule_length_20_seed_0_starting_epsilon_0.001_checkpoint.pth')['state_dict'])
    elif args.model == 'cifar': 
        train_loader, test_loader = pblm.cifar_loaders(args.batch_size)
        model = pblm.cifar_model().cuda()
        model.load_state_dict(torch.load('pixel2/cifar_small_batch_size_50_epochs_100_epsilon_0.0347_l1_proj_50_l1_test_median_l1_train_median_lr_0.05_momentum_0.9_opt_sgd_schedule_length_20_seed_0_starting_epsilon_0.001_weight_decay_0.0005_checkpoint.pth')['state_dict'])
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
        epsilon = Variable(args.epsilon*torch.ones(X.size(0)).cuda(), requires_grad=True)
        X, y = Variable(X).cuda(), Variable(y).cuda()

        out = Variable(model(X).data.max(1)[1])

        # form c without the 0 row
        c = Variable(torch.eye(num_classes).type_as(X.data)[out.data].unsqueeze(1) - torch.eye(num_classes).type_as(X.data).unsqueeze(0))
        I = (~(out.data.unsqueeze(1) == torch.arange(num_classes).type_as(out.data).unsqueeze(0)).unsqueeze(2))
        c = (c[I].view(X.size(0),num_classes-1,num_classes))
        if X.is_cuda:
            c = c.cuda()
        alpha = args.alpha

        def f(eps): 
            dual = DualNetBounds(model, X, eps.unsqueeze(1), True, True)
            f = -dual.g(c)
            return (f.max(1)[0])

        for i in range(args.niters): 
            f_max = f(epsilon)
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
            del f_max

        if i == args.niters - 1: 
            l.append(j)

        if  (y==out).data.sum() > 0: 
            correct.append(epsilon[y==out])
        if (y!=out).data.sum() > 0: 
            incorrect.append(epsilon[y!=out])

        del X, y
    print(l)
    torch.save(torch.cat(correct, 0), '{}_correct_epsilons.pth'.format(args.prefix))
    torch.save(torch.cat(incorrect, 0), '{}_incorrect_epsilons.pth'.format(args.prefix))