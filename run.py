import os
import subprocess
# uid, gid = os.system('id -u'), os.system('id -g')
uid = int(subprocess.run(['id', '-u'], stdout=subprocess.PIPE).stdout)
gid = int(subprocess.run(['id', '-g'], stdout=subprocess.PIPE).stdout)

gpu_id = 3

# name = 'mnist'
name = 'cifar'

epochs = 300
# epochs = 20
# epochs = 5
# epochs = 1

lr=0.05

# verbose = 0
verbose = 100

epsilon = 0.031
starting_epsilon=0.001

baseline = '--baseline'
# baseline = ''

# different model types
# model_type = ''
model_type = '--vgg'

s = """docker run -it --runtime=nvidia --rm -w /home -v ${{PWD}}/experiments/:/experiments/ convex_adversarial zsh -c '{}; {}; {}'"""
cmd = 'python examples/{0}.py --prefix /experiments/{0} {baseline} --lr {lr} --epochs {epochs} --verbose {verbose} {model_type} --epsilon {epsilon} --starting_epsilon {starting_epsilon}'.format(
    name, 
    lr=lr,
    verbose=verbose,
    epochs=epochs, 
    model_type=model_type,
    baseline=baseline,
    epsilon=epsilon,
    starting_epsilon=starting_epsilon
    )
set_gpu = 'export CUDA_VISIBLE_DEVICES={}'.format(gpu_id)
fix_permissions = 'chown -R {}:{} /experiments'.format(uid, gid)

full = s.format(set_gpu, cmd, fix_permissions)
os.system(full)