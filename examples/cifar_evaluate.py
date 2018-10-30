# import waitGPU
# import setGPU
# waitGPU.wait(utilization=20, available_memory=10000, interval=60)
# waitGPU.wait(gpu_ids=[1,3], utilization=20, available_memory=10000, interval=60)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random
    
import setproctitle

import problems as pblm
from trainer import *

import math
import numpy

def select_model(m): 
    if m == 'large': 
        # raise ValueError
        model = pblm.cifar_model_large().cuda()
    elif m == 'resnet': 
        model = pblm.cifar_model_resnet(N=1, factor=1).cuda()
    else: 
        model = pblm.cifar_model().cuda() 
    return model

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
numpy.random.seed(0)

if __name__ == "__main__": 
    args = pblm.argparser_evaluate(epsilon = 0.0347, norm='l1')

    print("saving file to {}".format(args.output))
    setproctitle.setproctitle(args.output)

    test_log = open(args.output, "w")

    _, test_loader = pblm.cifar_loaders(1)

    d = torch.load(args.load)

    model = []
    for sd in d['state_dict']: 
        m = select_model(args.model)
        m.load_state_dict(sd)
        model.append(m)

    best_err = 1

    epsilon = args.epsilon

    # robust cascade training
    err = evaluate_robust_cascade(test_loader, model,
       args.epsilon, 0, test_log, args.verbose,
       norm_type=args.norm, bounded_input=False, proj=args.proj)
