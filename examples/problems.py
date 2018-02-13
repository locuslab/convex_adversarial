import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import torch.utils.data as td

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def mnist_loaders(batch_size): 
    mnist_train = datasets.MNIST(".", train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST(".", train=False, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, pin_memory=False)
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
    return model

def mnist_model_large(): 
    model = nn.Sequential(
        nn.Conv2d(1, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(64*7*7,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
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