# import waitGPU
# import setGPU
# waitGPU.wait(utilization=50, available_memory=10000, interval=60)
# waitGPU.wait(gpu_ids=[1,3], utilization=20, available_memory=10000, interval=60)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
# cudnn.benchmark = True

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import setproctitle

import problems as pblm
from trainer import *
import math
import numpy as np

if __name__ == "__main__": 
    args = pblm.argparser(opt='adam', verbose=200, starting_epsilon=0.01)
    print("saving file to {}".format(args.prefix))
    setproctitle.setproctitle(args.prefix)
    if not args.eval:
        train_log = open(args.prefix + "_train.log", "w")
    test_log = open(args.prefix + "_test.log", "w")

    train_loader, test_loader = pblm.mnist_loaders(args.batch_size)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if args.model == 'vgg': 
        model = pblm.mnist_model_vgg().cuda()
        # s = 'experiments/mnist_vgg_proj/mnist200_vgg_batch_size_50_epochs_20_epsilon_0.001_l1_proj_200_l1_test_exact_l1_train_median_lr_0.001_opt_adam_seed_0_starting_epsilon_0.0001_model.pth'
        # model.load_state_dict(torch.load(s))
        # reduce the test set
        _, test_loader = pblm.mnist_loaders(1)
        # test_loader = [tl for i,tl in enumerate(test_loader) if i < 200]
    elif args.model == 'large': 
        model = pblm.mnist_model_large().cuda()
        # s = 'experiments/mnist_gradual/mnist2_large_batch_size_8_epochs_20_epsilon_0.1_l1_test_exact_l1_train_exact_lr_0.001_opt_adam_seed_0_starting_epsilon_0.1_model.pth'
        # s = 'experiments/mnist_large_madry.pth'
        # model.load_state_dict(torch.load(s))
        _, test_loader = pblm.mnist_loaders(8)
        # test_loader = [tl for i,tl in enumerate(test_loader) if i < 50]
    elif args.model == 'resnet': 
        model = pblm.mnist_model_resnet().cuda()
    elif args.model == 'bn': 
        model = pblm.mnist_model_bn().cuda()
    elif args.model == 'wide': 
        print("Using wide model with model_factor={}".format(args.model_factor))
        _, test_loader = pblm.mnist_loaders(64//args.model_factor)
        model = pblm.mnist_model_wide(args.model_factor).cuda()
    elif args.model == 'deep': 
        print("Using deep model with model_factor={}".format(args.model_factor))
        _, test_loader = pblm.mnist_loaders(64//(2**args.model_factor))
        model = pblm.mnist_model_deep(args.model_factor).cuda()
    else: 
        model = pblm.mnist_model().cuda() 
        #model.load_state_dict(torch.load('l1_truth/mnist_nonexact_rerun_baseline_False_batch_size_50_delta_0.01_epochs_20_epsilon_0.1_l1_proj_200_l1_test_exact_l1_train_median_lr_0.001_m_10_seed_0_starting_epsilon_0.05_model.pth'))

    for i, (X, y) in enumerate(test_loader): 
        X = Variable(X.cuda())
        y = Variable(y.cuda())
        break
    

    # for m in model.modules():
    #     if isinstance(m, nn.Conv2d):
    #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         m.weight.data.normal_(0, math.sqrt(2. / n))
    #         m.bias.data.zero_()

    # pblm.init_scale(model, X[:1], args.starting_epsilon)

    kwargs = pblm.args2kwargs(args, X=X)
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
        if args.opt == 'adam': 
            opt = optim.Adam(model.parameters(), lr=args.lr)
        elif args.opt == 'sgd': 
            opt = optim.SGD(model.parameters(), lr=args.lr, 
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
        else: 
            raise ValueError("Unknown optimizer")
        lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)
        # schedule epsilon over a maximum of 20 epochs
        eps_schedule = np.logspace(np.log10(args.starting_epsilon), 
                                   np.log10(args.epsilon), 
                                   args.schedule_length)
        for t in range(args.epochs):
            # reduce LR after epsilon schedule
            lr_scheduler.step(epoch=max(t-len(eps_schedule), 0))
            if args.method == 'baseline': 
                train_baseline(train_loader, model, opt, t, train_log, args.verbose)
                err = evaluate_baseline(test_loader, model, t, test_log,
                   args.verbose)
            elif args.method=='madry':
                train_madry(train_loader, model, args.epsilon, opt, t, train_log,
                   args.verbose)
                evaluate_madry(test_loader, model, args.epsilon, t, test_log,
                   args.verbose)
            else:
                if t < len(eps_schedule) and args.starting_epsilon is not None: 
                    # epsilon = args.starting_epsilon + (t/(args.epochs//2))*(args.epsilon - args.starting_epsilon)
                    epsilon = float(eps_schedule[t])
                else:
                    epsilon = args.epsilon
                train_robust(train_loader, model, opt, epsilon, t, train_log, 
                    args.verbose, l1_type=args.l1_train, bounded_input=True, 
                    **kwargs)
                err = evaluate_robust(test_loader, model, args.epsilon, t,
                   test_log,
                   args.verbose, l1_type=args.l1_test, bounded_input=True, 
                   **kwargs)
            
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
            # torch.save(model.state_dict(), args.prefix + "_model.pth")