import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import setGPU
import numpy as np
import cvxpy as cp

from convex_adversarial import DualNetBounds, robust_loss_batch

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import setproctitle
import argparse

def train_robust(loader, model, opt, epsilon, epoch, log):
    model.train()
    if epoch == 0:
        blank_state = opt.state_dict()

    for i, (X,y) in enumerate(loader):
        X,y = X.cuda(), y.cuda()

        robust_ce, robust_err = robust_loss_batch(model, epsilon, 
                                             Variable(X), Variable(y))
        out = model(Variable(X))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.data.max(1)[1] != y).float().sum()  / X.size(0)

        opt.zero_grad()
        robust_ce.backward()
        opt.step()

        print(epoch, i, robust_ce.data[0], robust_err, ce.data[0], err, file=log)
        print(epoch, i, robust_ce.data[0], robust_err, ce.data[0], err)
        log.flush()

        del X, y, robust_ce
        


def evaluate_robust(loader, model, epsilon, epoch, log):
    model.eval()
    for i, (X,y) in enumerate(loader):
        X,y = X.cuda(), y.cuda()
        robust_ce, robust_err = robust_loss_batch(model, epsilon, 
                                            Variable(X), Variable(y))
        out = model(Variable(X))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.data.max(1)[1] != y).float().sum()  / X.size(0)

        print(epoch, i, robust_ce.data[0], robust_err, ce.data[0], err, file=log)
        print(epoch, i, robust_ce.data[0], robust_err, ce.data[0], err)
        log.flush()
        del X, y, robust_ce


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument('--prefix')
    parser.add_argument('--alpha', default='default')
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


    # for X,y in test_loader:
    #     X = X.cuda()
    #     y = y.cuda()
    #     break
    # epsilon = 0.1
    # from convex_adversarial import robust_loss
    # ce_loss, ce_err = robust_loss(model, epsilon, Variable(X), Variable(y))
    # ce_loss.backward()

    opt = optim.Adam(model.parameters(), lr=args.lr)
    for t in range(args.epochs):
        train_robust(train_loader, model, opt, args.epsilon, t, train_log)
        evaluate_robust(test_loader, model, args.epsilon, t, test_log)
        torch.save(model.state_dict(), args.prefix + "_model.pth")