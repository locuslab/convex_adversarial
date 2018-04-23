import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
import torch.utils.data as td
import argparse
from convex_adversarial import epsilon_from_model, DualNetBounds
import math

def init_scale(model, X, epsilon): 
    dual = DualNetBounds(model, X, epsilon)
    for i in range(1, len(dual.affine)):
        d0 = (dual.zu[0] - dual.zl[0]).mean().data
        di = (dual.zu[i]-dual.zl[i]).mean().data
        while di[0] > d0[0]: 
            print("scaling by 0.5, ", i)
            dual.affine[i].l.weight.data *= 0.5
            del dual
            dual = DualNetBounds(model, X, epsilon)
            d0 = (dual.zu[0] - dual.zl[0]).mean().data
            di = (dual.zu[i]-dual.zl[i]).mean().data
    del dual

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def mnist_loaders(batch_size, shuffle_test=False): 
    mnist_train = datasets.MNIST(".", train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST(".", train=False, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=shuffle_test, pin_memory=False)
    return train_loader, test_loader

def fashion_mnist_loaders(batch_size): 
    mnist_train = datasets.MNIST("./fashion_mnist", train=True,
       download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST("./fashion_mnist", train=False,
       download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, pin_memory=False)
    return train_loader, test_loader

def mnist_model(): 
    model = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*7*7,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()
    return model

def mnist_model_large(): 
    model = nn.Sequential(
        nn.Conv2d(1, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(64*7*7,1024),
        nn.ReLU(),
        nn.Linear(1024, 10)
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()
    return model

def mnist_model_vgg(): 
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(64*7*7,512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512,10)
    )
    return model

def replace_10_with_0(y): 
    return y % 10

def svhn_loaders(batch_size): 
    train = datasets.SVHN(".", split='train', download=True, transform=transforms.ToTensor(), target_transform=replace_10_with_0)
    test = datasets.SVHN(".", split='test', download=True, transform=transforms.ToTensor(), target_transform=replace_10_with_0)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, pin_memory=False)
    return train_loader, test_loader

def svhn_model(): 
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*8*8,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    ).cuda()
    return model

def har_loaders(batch_size):     
    X_te = torch.from_numpy(np.loadtxt('../datasets/UCI HAR Dataset/test/X_test.txt')).float()
    X_tr = torch.from_numpy(np.loadtxt('../datasets/UCI HAR Dataset/train/X_train.txt')).float()
    y_te = torch.from_numpy(np.loadtxt('../datasets/UCI HAR Dataset/test/y_test.txt')-1).long()
    y_tr = torch.from_numpy(np.loadtxt('../datasets/UCI HAR Dataset/train/y_train.txt')-1).long()

    har_train = td.TensorDataset(X_tr, y_tr)
    har_test = td.TensorDataset(X_te, y_te)

    train_loader = torch.utils.data.DataLoader(har_train, batch_size=batch_size, shuffle=True, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(har_test, batch_size=batch_size, shuffle=False, pin_memory=False)
    return train_loader, test_loader

def har_500_model(): 
    model = nn.Sequential(
        nn.Linear(561, 500),
        nn.ReLU(),
        nn.Linear(500, 6)
    )
    return model

def har_500_250_model(): 
    model = nn.Sequential(
        nn.Linear(561, 500),
        nn.ReLU(),
        nn.Linear(500, 250),
        nn.ReLU(),
        nn.Linear(250, 6)
    )
    return model

def har_500_250_100_model(): 
    model = nn.Sequential(
        nn.Linear(561, 500),
        nn.ReLU(),
        nn.Linear(500, 250),
        nn.ReLU(),
        nn.Linear(250, 100),
        nn.ReLU(),
        nn.Linear(100, 6)
    )
    return model

def cifar_loaders(batch_size): 
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train = datasets.CIFAR10('.', train=True, download=True, 
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]))
    test = datasets.CIFAR10('.', train=False, 
        transform=transforms.Compose([transforms.ToTensor(), normalize]))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
        shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
        shuffle=False, pin_memory=True)
    return train_loader, test_loader

def cifar_model(): 
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*8*8,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

def cifar_model_vgg(): 
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 256, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(256, 256, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(256, 512, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(512, 512, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(512, 512, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(512, 512, 3, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512,10)
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()
    return model

class ResBlock(nn.Module): 
    def __init__(self, in_ch, out_ch, stride=1):
        super(ResBlock,self).__init__()
        if stride != 1:
            self.shortcut = nn.Conv2d(in_ch, out_ch,
                kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Sequential()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                                padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1,
                                padding=1)

    def forward(self, x): 
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

def cifar_model_resnet(): 
    model = nn.Sequential(
        nn.Conv2d(3,64, kernel_size=3, stride=1, padding=1),
        ResBlock(64, 64, stride=1),
        ResBlock(64, 128, stride=2), 
        ResBlock(128, 256, stride=2),
        ResBlock(256, 512, stride=2),
        Flatten(),
        nn.Linear(512*4*4,10)
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None: 
                m.bias.data.zero_()
    return model

def argparser(batch_size=50, epochs=20, seed=0, verbose=1, lr=1e-3, 
              epsilon=0.1, starting_epsilon=None, 
              l1_proj=None, delta=None, m=1, l1_eps=None, 
              l1_train='exact', l1_test='exact', 
              opt='sgd', momentum=0.9, weight_decay=5e-4): 

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--epochs', type=int, default=epochs)
    parser.add_argument('--seed', type=int, default=seed)
    parser.add_argument('--verbose', type=int, default=verbose)
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--epsilon", type=float, default=epsilon)
    parser.add_argument("--starting_epsilon", type=float, default=starting_epsilon)
    parser.add_argument('--prefix')
    parser.add_argument('--eval')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--alpha_grad', action='store_true')
    parser.add_argument('--scatter_grad', action='store_true')
    parser.add_argument('--l1_proj', type=int, default=l1_proj)
    parser.add_argument('--delta', type=float, default=delta)
    parser.add_argument('--m', type=int, default=m)
    parser.add_argument('--l1_eps', type=float, default=l1_eps)
    parser.add_argument('--large', action='store_true')
    parser.add_argument('--vgg', action='store_true')
    parser.add_argument('--resnet', action='store_true')
    parser.add_argument('--l1_train', default=l1_train)
    parser.add_argument('--l1_test', default=l1_test)
    parser.add_argument('--opt', default=opt)
    parser.add_argument('--momentum', type=float, default=momentum)
    parser.add_argument('--weight_decay', type=float, default=weight_decay)
    
    args = parser.parse_args()
    if args.starting_epsilon is None:
        args.starting_epsilon = args.epsilon 
    if args.prefix: 
        if args.vgg: 
            args.prefix += '_vgg'
        elif args.large: 
            args.prefix += '_large'
        elif args.resnet: 
            args.prefix += '_resnet'
        else:
            raise ValueError("unknown model type")

        if args.eval: 
            args.prefix += '_eval_' + args.eval.replace('/','_')
        else:
            banned = ['alpha_grad', 'scatter_grad', 'verbose', 'prefix',
                      'resume',
                      'large', 'vgg', 'resnet']
            if args.baseline:
                banned += ['epsilon', 'starting_epsilon', 'l1_test', 'l1_train', 'm', 'l1_proj']
            for arg in sorted(vars(args)): 
                if arg not in banned and getattr(args,arg) is not None: 
                    args.prefix += '_' + arg + '_' +str(getattr(args, arg))
    else: 
        args.prefix = 'mnist_temporary'

    return args

def args2kwargs(args, X=None): 

    if args.l1_proj is not None: 
        if not args.l1_eps:
            if args.delta: 
                args.l1_eps = epsilon_from_model(model, Variable(X.cuda()), args.l1_proj,
                                            args.delta, args.m)
                print('''
        With probability {} and projection into {} dimensions and a max
        over {} estimates, we have epsilon={}'''.format(args.delta, args.l1_proj,
                                                        args.m, args.l1_eps))
            else: 
                args.l1_eps = 0
                print('No epsilon or \delta specified, using epsilon=0.')
        else:
            print('Specified l1_epsilon={}'.format(args.l1_eps))
        kwargs = {
            'alpha_grad' : args.alpha_grad,
            'scatter_grad' : args.scatter_grad, 
            'l1_proj' : args.l1_proj, 
            'l1_eps' : args.l1_eps, 
            'm' : args.m
        }
    else:
        kwargs = {
            'alpha_grad' : args.alpha_grad,
            'scatter_grad' : args.scatter_grad, 
        }
    return kwargs