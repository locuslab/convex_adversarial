#!/bin/bash 
PYTHONPATH=/nethome/ericwong/convex_adversarial.preview.nips
# for L2 note that an L-infinity ball with radius eps
# has approximately the same volume as an L2 ball with radius
# sqrt(d/pi)*eps, where d is the number of dimensions. 
# For MNIST this is 
# sqrt(784/pi)*0.1=1.58
# sqrt(784/pi)*0.3=4.74
# and for cifar this is 
# sqrt(1024/pi)*0.0348=0.628
# sqrt(1024/pi)*0.139=2.51

# arguments that are universal across all experiments
cuda_ids=0,1,2,3
cascade=6
epochs=60
schedule_length=20

# L2 ball arguments
norm_type=l1_median
norm_eval=l1
# norm_type=l2_normal
# norm_eval=l2

# MNIST parameters
prefix="tmp/mnist"
starting_epsilon=0.01
parameters="--epochs ${epochs} --starting_epsilon ${starting_epsilon} --schedule_length ${schedule_length} --cascade ${cascade} --prefix ${prefix} --verbose 200 --cuda_ids ${cuda_ids}"

# [pick an epsilon]
# linf ball epsilons
eps=0.1
# eps=0.3
# l2 ball epsilons
# eps=1.58

# small, exact 
python examples/mnist.py --epsilon ${eps} --norm_train ${norm_eval} --norm_test ${norm_eval} ${parameters}

# all remaining experiments use an approximation for training with 50 projections
parameters="--proj 50 --norm_train ${norm_type} ${parameters}"

# small 
# python examples/mnist.py --epsilon ${eps} --norm_test ${norm_eval} ${parameters} 

# large
# python examples/mnist.py --epsilon ${eps} --norm_test ${norm_eval} --test_batch_size 8 --model large ${parameters}

# CIFAR parameters
prefix="tmp/cifar"
starting_epsilon=0.001
parameters="--epochs ${epochs} --starting_epsilon ${starting_epsilon} --schedule_length ${schedule_length} --cascade ${cascade} --prefix ${prefix} --verbose 200 --cuda_ids ${cuda_ids}"
parameters="--proj 50 --norm_train ${norm_type} ${parameters}"

# [pick an epsilon]
# linf ball epsilons
eps=0.0348
# eps=0.139
# l2 ball epsilons
# eps=0.157

# small
# python examples/cifar.py --epsilon ${eps} --norm_test ${norm_eval} --test_batch_size 25 ${parameters}

# large
# python examples/cifar.py --epsilon ${eps} --norm_test ${norm_eval} --test_batch_size 8 --model large ${parameters}

# resnet
# the only model where we defer true (non estimated) test statistics 
# to after training to save time
# python examples/cifar.py --epsilon ${eps} --norm_test ${norm_type} --model resnet ${parameters}
