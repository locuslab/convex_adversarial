import waitGPU
waitGPU.wait(utilization=20, available_memory=10000, interval=10)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from convex_adversarial import robust_loss_batch

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import setproctitle
import argparse

from trainer import *

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--starting_epsilon", type=float, default=None)
    parser.add_argument('--prefix')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--alpha_grad', action='store_true')
    parser.add_argument('--scatter_grad', action='store_true')
    parser.add_argument('--old_weights', action='store_true')
    parser.add_argument('--l1_proj', type=int, default=None)
    args = parser.parse_args()
    args.prefix = args.prefix or 'mnist_conv_{:.4f}_{:.4f}_0'.format(args.epsilon, args.lr).replace(".","_")
    setproctitle.setproctitle(args.prefix)

    train_log = open(args.prefix + "_train.log", "w")
    test_log = open(args.prefix + "_test.log", "w")

    mnist_train = datasets.MNIST(".", train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST(".", train=False, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=args.batch_size, shuffle=True, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=args.batch_size, shuffle=False, pin_memory=False)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    model = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*7*7,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    ).cuda()
    
    if args.old_weights: 
        model.load_state_dict(torch.load('torch_0_2_weights.pth'))

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
                args.verbose, 
                args.alpha_grad, args.scatter_grad, l1_proj=args.l1_proj)
            evaluate_robust(test_loader, model, args.epsilon, t, test_log, args.verbose)

        torch.save(model.state_dict(), args.prefix + "_model.pth")