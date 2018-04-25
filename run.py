import os
import subprocess
# uid, gid = os.system('id -u'), os.system('id -g')
subprocess.run(['docker', 'build',  '-qt', 'convex_adversarial', '.'])

uid = int(subprocess.run(['id', '-u'], stdout=subprocess.PIPE).stdout)
gid = int(subprocess.run(['id', '-g'], stdout=subprocess.PIPE).stdout)

# gpu_id = 0
# gpu_id = 1
# gpu_id = 2
gpu_id = 2

# mnist-vgg exact
# name = 'mnist'
# epochs = 20
# lr=0.001
# epsilon = 0.01
# starting_epsilon=1e-3
# verbose = 100
# batch_size=4
# model_type='--vgg'
# baseline=''
# extra = ''

# baseline cifar-vgg
# name = 'cifar'
# epochs = 300
# lr=0.05
# verbose = 20
# epsilon = 0.031
# starting_epsilon=0.001
# batch_size=128
# baseline = '--baseline'
# model_type = '--vgg_pytorch'
# extra = ''

# baseline cifar-resnet
name = 'cifar'
epochs = 300
lr=0.01
verbose = 100
epsilon = 0.031
starting_epsilon=0.001
batch_size=128
baseline = '--baseline'
model_type = '--resnet_bn'
extra = ''



# baseline = ''

# different model types
# model_type = '--resnet'



s = """docker run -it --runtime=nvidia --rm -w /home -v ${{PWD}}/experiments/:/experiments/ convex_adversarial zsh -c '{}; {}; {}'"""
cmd = 'python examples/{0}.py --prefix /experiments/{0}{extra} {baseline} --lr {lr} --epochs {epochs} --verbose {verbose} {model_type} --epsilon {epsilon} --starting_epsilon {starting_epsilon} --batch_size {batch_size}'.format(
    name, 
    lr=lr,
    verbose=verbose,
    epochs=epochs, 
    model_type=model_type,
    baseline=baseline,
    epsilon=epsilon,
    starting_epsilon=starting_epsilon,
    extra=extra, 
    batch_size=batch_size
    )
set_gpu = 'export CUDA_VISIBLE_DEVICES={}'.format(gpu_id)
fix_permissions = 'chown -R {}:{} /experiments'.format(uid, gid)

full = s.format(set_gpu, cmd, fix_permissions)
os.system(full)