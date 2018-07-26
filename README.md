# Provably robust neural networks

*A repository for training provably robust neural networks by optimizing convex outer bounds on the adversarial polytope. Created by [Eric Wong](https://riceric22.github.io) and [Zico Kolter](http://zicokolter.com). [Link to the original arXiv paper][paper]. The method has been further extended to be fully modular, scalable, and use cascades to improve robust error. Check out our new paper on arXiv: [Scaling provable adversarial defenses][scalable_paper].*

[paper]: https://arxiv.org/abs/1711.00851
[scalable_paper]: https://arxiv.org/abs/1805.12514

## News
+ 7/26/2018 - Version 0.3.3 code refactor
+ 6/2/2018 - Version 0.3.1 released to reflect the new paper. 
+ 5/31/2018 - New paper on a scalable version for models with skip connections
and a fully modular implementation for simple extension. Code base with these
improvements with a port to PyTorch 0.4 will be released shortly. 
+ 4/26/2018 - Added robust models from the paper to the `models/` folder in the
repository. 
+ 3/4/2018 - Updated paper with more experiments. Code migrated to work with
  PyTorch 0.3.0+. Real mini-batching implemented, with a 5x speedup over the
  old codebase, and several NaN bugfixes. 
+ 12/8/2017 - Best defense paper at the NIPS 2017 ML & Security Workshop
+ 11/2/2017 - Initial preprint and release of codebase. 

## Installation & Usage
You can install this repository with 
`pip install convex_adversarial` 

If you wish to have the version of code that reflects the first paper, use 
`pip install convex_adversal=0.2`, or clone the [0.2 release on Github](https://github.com/locuslab/convex_adversarial/tree/v0.2). 

The package contains the following functions: 
+ `robust_loss(net, epsilon, X, y, l1_proj=None,
                 l1_type='exact', bounded_input=False, size_average=True)`
    computes a robust loss function for a given ReLU network `net` and l1 
    radius `epsilon` for examples `X` and their labels `y`. You can use 
    this as a drop in replacement for, say, `nn.CrossEntropyLoss()`, and is
    equivalent to the objective of Equation 14 in the original paper. 
    To use the scalable version, specify a projection dimension with `l1_proj`
    and set `l1_type` to `median`. 
+ `robust_loss_parallel` computes the same objective as `robust_loss`, but
    only for a *single* example and using  
    data parallelism. This is useful for exact evaluation if a single 
    example doesn't fit in memory. 
+ `dual_net = DualNetwork(net, X, epsilon, l1_proj=None, l1_type='exact', bounded_input=False)`
    is a PyTorch module that computes the layer-wise upper and lower bounds for
    all activations in the network. This is useful if you are only interested 
    in the bounds and not the robust loss, and corresponds to Algorithm 
    1 in the paper. 
+ `dual_net(c)` is the module's forward pass which computes the lower
    bound on the primal problem described in the paper for a given 
    objective vector c. This corresponds to computing objective of Theorem 1 in
    the paper (Equation 5). 

## Why do we need robust networks? 
While networks are capable of representing highly complex functions. For
example, with today's networks it is an easy task to achieve 99% accuracy on
the MNIST digit recognition dataset, and we can quickly train a small network
that can accurately predict that the following image is a 7.

<img src="https://github.com/locuslab/convex_adversarial.release/blob/master/images/seven.png" width="100">

However, the versatility of neural networks comes at a cost: these networks
are highly susceptible to small perturbations, or adversarial attacks (e.g. the [fast gradient sign method](https://arxiv.org/abs/1412.6572) and [projected gradient descent](https://arxiv.org/abs/1706.06083))! While
most of us can recognize that the following image is still a 7, the same
network that could correctly classify the above image instead classifies 
the following image as a 3.

<img src="https://github.com/locuslab/convex_adversarial.release/blob/master/images/seven_adversarial.png" width="100">

While this is a relatively harmless example, one can easily think of
situations where such adversarial perturbations can be dangerous and costly
(e.g. autonomous driving). 

## What are robust networks? 
Robust networks are networks that are trained to protect against any sort of
adversarial perturbation. Specifically, for any seen training example, the
network is robust if it is impossible to cause the network to incorrectly
classify the example by adding a small perturbation.

## How do we do this? 
The short version: we use the dual of a convex relaxation of the network over
the adversarial polytope to lower bound the output. This lower bound can be
expressed as another deep network with the same model parameters, and
optimizing this lower bound allows us to guarantee robustness of the network.

The long version: see our original paper, 
[Provable defenses against adversarial examples via the convex outer adversarial polytope][paper]. 

For our updated version which is scalable, modular, and achieves even better 
robust performance, see our new paper, 
[Scaling provable adversarial defenses][scalable_paper]. 

## What difference does this make? 
We illustrate the power of training robust networks in the following two scenarios: 2D toy case for a visualization, and on the MNIST dataset. More experiments are in the paper. 

### 2D toy example
To illustrate the difference, consider a binary classification task on 2D
space, separating red dots from blue dots. Optimizing a neural network in the
usual fashion gives us the following classifier on the left, and our robust
method gives the classifier on the right. The squares around each example
represent the adversarial region of perturbations.

<p float="left">
<img src="https://github.com/locuslab/convex_adversarial.release/blob/master/images/normal_trained.png" width="300">
<img src="https://github.com/locuslab/convex_adversarial.release/blob/master/images/robust_trained.png" width="300">
</p>

For the standard classifier, a number of the examples have perturbation
regions that contain both red and blue. These examples are susceptible to
adversarial attacks that will flip the output of the neural network. On the
other hand, the robust network has all perturbation regions fully contained in
the either red or blue, and so this network is robust: we are guaranteed that
there is no possible adversarial perturbation to flip the label of any
example.

### Robustness to adversarial attacks: MNIST classification
As mentioned before, it is easy to fool networks trained on the MNIST dataset 
when using attacks such as the fast gradient sign method (FGS) and projected gradient descent (PGD). We observe that PGD can almost always fool the MNIST trained network. 

|          | Base error | FGS error | PGD Error | Robust Error |
| --------:| ----------:|----------:| ---------:| ------------:|
| Original |       1.1% |     50.0% |     81.7% |         100% |
|   Robust |       1.8% |      3.9% |      4.1% |         5.8% |

On the other hand, the robust network is significantly less affected by these
attacks. In fact, when optimizing the robust loss, we can additionally
calculate a *robust error* which gives an provable upper bound on the error
caused by *any* adversarial perturbation. In this case, the robust network has
a robust error of 5.8%, and so we are guaranteed that no adversarial attack
can ever get an error rate of larger than 5.8%. In comparison, the robust
error of the standard network is 100%. More results on HAR, Fashion-MNIST, and
SVHN can be found in the [paper][paper]. Results for the scalable version with
random projections on residual networks and on the CIFAR10 dataset can be found
in our [second paper][scalable_paper].

## Modularity

### Dual operations
The package currently has dual operators for the following constrained input
spaces and layers. These are defined in `dual_inputs.py` and `dual_layers.py`. 

#### Dual input spaces
+ `InfBall` : L-infinity ball constraint on the input
+ `InfBallBounded` : L-infinity ball constraint on the input, with additional
bounding box constraints (works for [0,1] box constraints). 
+ `InfBallProj` : L-infinity ball constraint using Cauchy random projections
+ `InfBallProjBounded` : L-infinity ball constraint using Cauchy random
projections, with additional bounding box constraints (works for [0,1] box
constraints)

#### Dual layers
+ `DualLinear` : linear, fully connected layers
+ `DualConv2d` : 2d convolutional layers
+ `DualReshape` : reshaping layers, e.g. flattening dimensions
+ `DualReLU` : ReLU activations
+ `DualReLUProj` : ReLU activations using Cauchy random projections
+ `DualDense` : Dense layers, for skip connections
+ `DualBatchNorm2d` : 2d batch-norm layers, assuming a fixed mean and variance
+ `Identity` : Identity operator, e.g. for some ResNet skip connections

Due to the modularity of the implementation, it is easy to extend the
methodology to additional dual layers. A dual input or dual layer can be
implemented by filling in the following signature: 

```python
class DualObject(nn.Module, metaclass=ABCMeta): 
    @abstractmethod
    def __init__(self): 
        """ Initialize a dual layer by initializing the variables needed to
        compute this layer's contribution to the upper and lower bounds. 

        In the paper, if this object is at layer i, this is initializing `h'
        with the required cached values when nu[i]=I and nu[i]=-I. 
        """pass

    @abstractmethod
    def apply(self, dual_layer):
        """ Advance cached variables initialized in this class by the given
        dual layer.  """
        raise NotImplementedError

    @abstractmethod
    def bounds(self): 
        """ Return this layers contribution to the upper and lower bounds. In
        the paper, this is the `h' upper bound where nu is implicitly given by
        c=I and c=-I. """
        raise NotImplementedError

    @abstractmethod
    def objective(self, *nus): 
        """ Return this layers contribution to the objective, given some
        backwards pass. In the paper, this is the `h' upper bound evaluated on a
        the given nu variables. 

        If this is layer i, then we get as input nu[k] through nu[i]. 
        So non-residual layers will only need nu[-1] and nu[-2]. """
        raise NotImplementedError

class DualLayer(DualObject): 
    @abstractmethod
    def forward(self, *xs): 
        """ Given previous inputs, apply the affine layer (forward pass) """ 
        raise NotImplementedError

    @abstractmethod
    def T(self, *xs): 
        """ Given previous inputs, apply the transposed affine layer 
        (backward pass) """
        raise NotImplementedError
```

## Residual networks / skip connections

To create sequential PyTorch modules with skip connections, we provide a
generalization of the PyTorch module `nn.Sequential`. Specifically, we have a
`DenseSequential` module that is identical to `nn.Sequential` but also takes
in `Dense' modules. The `Dense' modules consist of `m` layers, and applies
these `m` layers to the last `m` outputs of the network. 

As an example, the
following is a simple two layer network with a single skip connection. 
The first layer is identical to a normal `nn.Conv2d` layer. The second layer has
a skip connection from the layer with 16 filters and also a normal convolutional
layer from the previous layer with 32 filters. 

```python
residual_block = DenseSequential([
    Dense(nn.Conv2d(16,32,...)),
    nn.ReLU(), 
    Dense(nn.Conv2d(16,32,...), None, nn.Conv2d(32,32,...))
])
```

## What is in this repository? 
+ The code implementing the robust loss function that measures the convex
  outer bounds on the adversarial polytope as described in the paper. 
+ The implemented dual layers are linear layers, convolutional layers, ReLU
  layers, ReLU layers with random projections, all of which can be used with
  skip connections. 
+ The implemented dual input constraints are l1 bounded perturbations, its
projection variant, and l1 bounded perturbation with an additional [0,1]
rectangle box constraint. 
+ Examples, containing the following: 
  + Code to train a robust classifier for the MNIST, Fashion-MNIST, HAR, and SVHN datasets. 
  + Code to generate and plot the 2D toy example.
  + Code to find minimum distances to the decision boundary of the neural network
  + Code to attack models using FGS and PGD
  + Code to solve the primal problem exactly using CVXPY
  + Code to train large MNIST and CIFAR10 models using random projections and
  residual networks