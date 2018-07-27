import torch.nn as nn

from abc import ABCMeta, abstractmethod

class DualObject(nn.Module, metaclass=ABCMeta): 
    def __init__(self): 
        """ Initialize a dual layer by initializing the variables needed to
        compute this layer's contribution to the upper and lower bounds. 

        In the paper, if this object is at layer i, this is initializing `h'
        with the required cached values when nu[i]=I and nu[i]=-I. 
        """
        super(DualObject, self).__init__()

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
