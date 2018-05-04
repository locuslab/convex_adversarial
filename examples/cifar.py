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

import setproctitle

import problems as pblm
from trainer import *

if __name__ == "__main__": 
    args = pblm.argparser(epsilon = 0.031, batch_size = 32)
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
    torch.cuda.manual_seed(args.seed)


    if args.model == 'vgg': 
        # raise ValueError
        model = pblm.cifar_model_vgg().cuda()
        _, test_loader = pblm.cifar_loaders(1, )
        test_loader = [tl for i,tl in enumerate(test_loader) if i < 200]
    elif args.model == 'resnet': 
        model = pblm.cifar_model_resnet(N=1, factor=1).cuda()
        #model = pblm.mnist_model_large().cuda()

    elif args.model == 'wide': 
        model = pblm.cifar_model_wide().cuda()
    else: 
        model = pblm.cifar_model().cuda() 
        #model.load_state_dict(torch.load('l1_truth/mnist_nonexact_rerun_baseline_False_batch_size_50_delta_0.01_epochs_20_epsilon_0.1_l1_proj_200_l1_test_exact_l1_train_median_lr_0.001_m_10_seed_0_starting_epsilon_0.05_model.pth'))

    starting_epoch=0
    if args.resume: 
        checkpoint = torch.load(args.prefix + '_checkpoint.pth')
        model.load_state_dict(checkpoint['state_dict'])
        starting_epoch = checkpoint['epoch']+1

    kwargs = pblm.args2kwargs(args)
    best_err = 1
    best_state_dict = model.state_dict()

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
        lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.5)
        eps_schedule = np.logspace(np.log10(args.starting_epsilon), 
                                   np.log10(args.epsilon), 
                                   args.epochs//2)
        for t in range(starting_epoch, args.epochs):
            lr_scheduler.step(epoch=t)
            if args.method == 'baseline': 
                train_baseline(train_loader, model, opt, t, train_log, args.verbose)
                err = evaluate_baseline(test_loader, model, t, test_log, args.verbose)
            else:
                if t < args.epochs//2 and args.starting_epsilon is not None: 
                    # epsilon = args.starting_epsilon + (t/(args.epochs//2))*(args.epsilon - args.starting_epsilon)
                    epsilon = float(eps_schedule[t])
                else:
                    epsilon = args.epsilon
                train_robust(train_loader, model, opt, epsilon, t, train_log, 
                    args.verbose, l1_type=args.l1_train, **kwargs)
                err = evaluate_robust(test_loader, model, args.epsilon, t, test_log,
                   args.verbose, l1_type=args.l1_test, **kwargs)
            print('Epoch {}: {} err'.format(t, err))
            
            if err < best_err: 
                best_state_dict = model.state_dict()
                best_err = err
                
            torch.save({ 
                'state_dict': model.state_dict(),
                'err' : err,
                'best_state_dict' : best_state_dict, 
                'best_err' : best_err,
                'epoch' : t
                }, args.prefix + "_checkpoint.pth")