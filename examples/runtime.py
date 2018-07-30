import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import gpustat
import numpy as np

from problems import Flatten

def gpu_mem(): 
    stats = gpustat.GPUStatCollection.new_query()
    for gpu in stats: 
        util = gpu.entry['memory.used']
        break
    return util

# random points at least 2r apart
m = 10
np.random.seed(3)
x = [np.random.uniform(size=(1,28,28))]
r = 0.16
while(len(x) < m):
    p = np.random.uniform(size=(1,28,28))
    if min(np.abs(p-a).sum() for a in x) > 2*r:
        x.append(p)
# r = 0.145
epsilon = r/2

X = torch.Tensor(np.array(x)).cuda()
torch.manual_seed(1)
y = (torch.rand(m)+0.5).long().cuda()

import sys
sys.path.append("../")
from convex_adversarial import robust_loss

import time

class Meter: 
    def __init__(self): 
        self.l = [[]]
    def add(self, x): 
        self.l[-1].append(x)
    def next(self): 
        self.l.append([])
    def save(self, fname): 
        x = np.array(self.l[:-1])

        np.savetxt(fname, x)

xs, ys = Meter(), Meter()
mems = Meter()

PROJ = True

for j in range(1,1001): 
    try: 
        for _ in range(10): 
            torch.cuda.empty_cache()
            start_mem = gpu_mem()

            # torch.manual_seed(1)
            robust_net = nn.Sequential(
                nn.Conv2d(1, j, 3, stride=1, padding=1),
                nn.ReLU(),
                Flatten(), 
                nn.Linear(j*28*28,2)
            ).cuda()
            data = []
            opt = optim.Adam(robust_net.parameters(), lr=1e-3)


            ts = []

            for i in range(10):
                start_time = time.time()
                if PROJ: 
                    robust_ce, robust_err = robust_loss(robust_net, epsilon, X, y,
                    parallel=False, l1_proj=50, l1_type='median')
                else:
                    robust_ce, robust_err = robust_loss(robust_net, epsilon, X, y,
                    parallel=False)

                out = robust_net(X)
                l2 = nn.CrossEntropyLoss()(out, y).item()
                err = (out.max(1)[1] != y).float().mean().item()
                data.append([l2, robust_ce.item(), err, robust_err])
                # if i % 100 == 0:
                #     print(robust_ce.item(), robust_err)
                opt.zero_grad()
                (robust_ce).backward()
                opt.step()

                end_time = time.time()
                ts.append(end_time-start_time)

            end_mem = gpu_mem()
            mems.add(end_mem)
            # print(start_mem, end_mem)
            del robust_net, robust_ce, l2, robust_err, err, out, opt
            # print(globals().keys())
            # assert False

            # print(l2, robust_ce.item(), robust_err)
            ts = np.array(ts[1:])
            xs.add(j*100*4)
            ys.add(ts.mean())
            print(j*28*28, ts.mean(), end_mem)
        mems.next()
        xs.next()
        ys.next()
    except: 
        break
if PROJ: 
    xs.save('sizes_conv_proj.txt')
    ys.save('epoch_conv_times_proj.txt')
    mems.save('memory_conv_proj.txt')
else:
    xs.save('sizes_conv_full.txt')
    ys.save('epoch_conv_times_full.txt')
    mems.save('memory_conv_full.txt')