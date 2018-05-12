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
# torch.backends.cudnn.deterministic = True

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random
    
import setproctitle

import problems as pblm
from trainer import *

import math
import numpy

if __name__ == "__main__": 
    args = pblm.argparser(epsilon = 0.139, batch_size = 50, lr=0.05)
    # args = pblm.argparser(epsilon = 0.031, batch_size = 32, lr=0.05)
    print("saving file to {}".format(args.prefix))
    setproctitle.setproctitle(args.prefix)

    if not args.eval:
        train_log = open(args.prefix + "_train.log", "w")
    test_log = open(args.prefix + "_test.log", "w")

    train_loader, test_loader = pblm.cifar_loaders(args.batch_size)
    # if args.vgg: 
    #     _, test_loader = pblm.mnist_loaders(1, shuffle_test=True)
    #     test_loader = [tl for i,tl in enumerate(test_loader) if i < 200]

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(0)
    numpy.random.seed(0)

    if args.model == 'vgg': 
        # raise ValueError
        model = pblm.cifar_model_vgg().cuda()
        if args.l1_test == 'exact': 
            _, test_loader = pblm.cifar_loaders(1, shuffle_test=True)
            test_loader = [tl for i,tl in enumerate(test_loader) if i < 1000]
    elif 'resnet' in args.model: 
        model = pblm.cifar_model_resnet(N=args.resnet_N, args.resnet_factor).cuda()
        if args.l1_test == 'exact': 
            _, test_loader = pblm.cifar_loaders(1, shuffle_test=True)
            test_loader = [tl for i,tl in enumerate(test_loader) if i < 1000]
        #model = pblm.mnist_model_large().cuda()

    elif args.model == 'wide': 
        model = pblm.cifar_model_wide(args.model_factor).cuda()
    elif args.model == 'deep': 
        model = pblm.cifar_model_deep(args.model_factor).cuda()
    else: 
        model = pblm.cifar_model().cuda() 
        #model.load_state_dict(torch.load('l1_truth/mnist_nonexact_rerun_baseline_False_batch_size_50_delta_0.01_epochs_20_epsilon_0.1_l1_proj_200_l1_test_exact_l1_train_median_lr_0.001_m_10_seed_0_starting_epsilon_0.05_model.pth'))

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()

    starting_epoch=0
    if args.resume: 
        checkpoint = torch.load(args.prefix + '_checkpoint.pth')
        model.load_state_dict(checkpoint['state_dict'])
        starting_epoch = checkpoint['epoch']+1

    kwargs = pblm.args2kwargs(args)
    best_err = 1

    if args.eval is not None: 
        try: 
            model.load_state_dict(torch.load(args.eval))
        except:
            print('[Warning] eval argument could not be loaded, evaluating a random model')
        evaluate_robust(test_loader, model, args.epsilon, 0, test_log,
            args.verbose, 
              **kwargs)
    else: 
        # opt = optim.Adam(model.parameters(), lr=args.lr)
        opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
            weight_decay=args.weight_decay)
        lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)
        eps_schedule = np.logspace(np.log10(args.starting_epsilon), 
                                   np.log10(args.epsilon), 
                                   args.schedule_length)
        for t in range(starting_epoch, args.epochs):
            lr_scheduler.step(epoch=max(t-len(eps_schedule), 0))
            if args.method == 'baseline': 
                train_baseline(train_loader, model, opt, t, train_log, args.verbose)
                err = evaluate_baseline(test_loader, model, t, test_log, args.verbose)
            else:
                if t < len(eps_schedule) and args.starting_epsilon is not None: 
                    # epsilon = args.starting_epsilon + (t/(args.epochs//2))*(args.epsilon - args.starting_epsilon)
                    epsilon = float(eps_schedule[t])
                else:
                    epsilon = args.epsilon
                train_robust(train_loader, model, opt, epsilon, t, train_log, 
                    args.verbose, l1_type=args.l1_train, clip_grad=1, **kwargs)
                err = evaluate_robust(test_loader, model, args.epsilon, t, test_log,
                   args.verbose, l1_type=args.l1_test, **kwargs)
            print('Epoch {}: {} err'.format(t, err))
            
            if err < best_err: 
                best_err = err
                torch.save({
                    'state_dict' : model.state_dict, 
                    'err' : best_err,
                    'epoch' : t
                    }, args.prefix + "_best.pth")
                
            torch.save({ 
                'state_dict': model.state_dict(),
                'err' : err,
                'epoch' : t
                }, args.prefix + "_checkpoint.pth")