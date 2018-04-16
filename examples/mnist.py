import waitGPU
import setGPU
# waitGPU.wait(utilization=20, available_memory=10000, interval=60)
# waitGPU.wait(gpu_ids=[1,3], utilization=20, available_memory=10000, interval=60)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import setproctitle

import problems as pblm
from trainer import *

if __name__ == "__main__": 
    args = pblm.argparser()
    print("saving file to {}".format(args.prefix))
    setproctitle.setproctitle(args.prefix)

    if not args.eval:
        train_log = open(args.prefix + "_train.log", "w")
    test_log = open(args.prefix + "_test.log", "w")

    train_loader, test_loader = pblm.mnist_loaders(args.batch_size)
    if args.vgg: 
        _, test_loader = pblm.mnist_loaders(1, shuffle_test=True)
        test_loader = [tl for i,tl in enumerate(test_loader) if i < 200]

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if args.vgg: 
        model = pblm.mnist_model_vgg().cuda()
    elif args.large: 
        model = pblm.mnist_model_large().cuda()
    else: 
        model = pblm.mnist_model().cuda() 
        #model.load_state_dict(torch.load('l1_truth/mnist_nonexact_rerun_baseline_False_batch_size_50_delta_0.01_epochs_20_epsilon_0.1_l1_proj_200_l1_test_exact_l1_train_median_lr_0.001_m_10_seed_0_starting_epsilon_0.05_model.pth'))

    kwargs = pblm.args2kwargs(args)

    if args.eval is not None: 
        try: 
            model.load_state_dict(torch.load(args.eval))
        except:
            print('[Warning] eval argument could not be loaded, evaluating a random model')
        evaluate_robust(test_loader, model, args.epsilon, 0, test_log,
            args.verbose, 
              **kwargs)
    else: 
        opt = optim.Adam(model.parameters(), lr=args.lr)
        eps_schedule = np.logspace(np.log10(args.starting_epsilon), 
                                   np.log10(args.epsilon), 
                                   args.epochs//2)
        for t in range(args.epochs):
            if args.baseline: 
                train_baseline(train_loader, model, opt, t, train_log, args.verbose)
                evaluate_baseline(test_loader, model, t, test_log, args.verbose)
            else:
                if t < args.epochs//2 and args.starting_epsilon is not None: 
                    # epsilon = args.starting_epsilon + (t/(args.epochs//2))*(args.epsilon - args.starting_epsilon)
                    epsilon = float(eps_schedule[t])
                else:
                    epsilon = args.epsilon
                train_robust(train_loader, model, opt, epsilon, t, train_log, 
                    args.verbose, l1_type=args.l1_train, **kwargs)
                evaluate_robust(test_loader, model, args.epsilon, t, test_log,
                   args.verbose, l1_type=args.l1_test, **kwargs)
                      # l1_geometric=args.l1_proj)

            torch.save(model.state_dict(), args.prefix + "_model.pth")