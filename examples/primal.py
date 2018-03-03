import setGPU
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from convex_adversarial import DualNetBounds, Affine, full_bias

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import setproctitle
import argparse

import problems as pblm
import cvxpy as cp

import numpy as np

cp2np = lambda x : np.asarray(x.value).T

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--niters', type=int, default=100)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=1)
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
        # model.load_state_dict(torch.load('icml/mnist_epochs_100_baseline_model.pth'))
        model.load_state_dict(torch.load('icml/mnist_epochs100_model.pth'))
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

    epsilon = 0.1
    num_classes = model[-1].out_features
    log = open(args.prefix + "_primal.log", "w")
    
    loader = train_loader if args.train else test_loader

    for j,(X,y) in enumerate(loader): 
        print('*** Batch {} ***'.format(j))
        dual = DualNetBounds(model, Variable(X.cuda()), epsilon, True, True)
        C = torch.eye(num_classes).type_as(X)[y].unsqueeze(1) - torch.eye(num_classes).type_as(X).unsqueeze(0)

        upper_bound = -dual.g(Variable(C.cuda())).data

        layers = dual.layers
        affine = dual.affine
        k = len(layers)

        W = [l(Variable(torch.eye(l.in_features).cuda())).t().cpu().data.numpy() for l in affine]
        b = [bias.view(-1).cpu().data.numpy() for bias in dual.biases]

        for i0,x in enumerate(X.numpy()): 
            # if i0 == 0: 
            #     continue
            print('*** Example {} ***'.format(i0))

            x = x.reshape(-1)
            zl = [l.data[i0,:].cpu().numpy() for l in dual.zl]
            zu = [u.data[i0,:].cpu().numpy() for u in dual.zu]
            I_minus = [u < 0 for u in zu]
            I_plus = [l > 0 for l in zl]
            I = [(u >= 0) * (l <= 0) for u,l in zip(zu,zl)]

            primal_values = []
            for j0,c in enumerate(C[i0].numpy()):
                z = [cp.Variable(l.in_features) for l in affine]
                zhat = [cp.Variable(l.out_features) for l in affine]


                cons_eq = [zhat[i] == W[i]*z[i] + b[i] for i in range(k)]
                cons_ball = [z[0] >= x - epsilon, z[0] <= x + epsilon]
                cons_zero = [z[i] >= 0 for i in range(1,k)]
                cons_linear = [z[i+1] >= zhat[i] for i in range(k-1)]
                cons_upper = [(cp.mul_elemwise(-(np.maximum(zu[i],0) - np.maximum(zl[i], 0)), zhat[i]) +
                               cp.mul_elemwise((zu[i] - zl[i]), z[i+1]) <= 
                               zu[i]*np.maximum(zl[i],0) - zl[i]*np.maximum(zu[i],0)) for i in range(k-1)]
                
                cons = cons_eq + cons_ball + cons_zero + cons_linear + cons_upper
                fobj = cp.Problem(cp.Minimize(c*zhat[-1]), cons).solve(verbose=False)

                print(i0, j0, upper_bound[i0][j0], -fobj)
                print(i0, j0, upper_bound[i0][j0], -fobj, file=log)