# written by Shuhei Horiguchi

from .base import EvaluatorBase
from ...util.general import samples_multidimensional_uniform
import sampyl as smp
from sampyl import np
#import numpy as np
import scipy.optimize
from sklearn.cluster import KMeans

class KMBBO(EvaluatorBase):
    """
    Class for the batch method on 'Efficient and Scalable Batch Bayesian Optimization Using K-Means' (Groves et al., 2018).

    :param acquisition: acquisition function to be used to compute the batch.
    :param batch size: the number of elements in the batch.

    """
    def __init__(self, acquisition, batch_size, N_sample=200, warmup=100):
        super(KMBBO, self).__init__(acquisition, batch_size)
        self.acquisition = acquisition
        self.batch_size = batch_size
        self.N_sample = N_sample
        self.warmup = warmup

    def compute_batch(self, duplicate_manager=None, context_manager=None, batch_context_manager=None):
        """
        Computes the elements of the batch.
        """
        assert not batch_context_manager or len(batch_context_manager) == self.batch_size
        if batch_context_manager:
            self.acquisition.optimizer.context_manager = batch_context_manager[0]
            raise NotImplementedError("batch_context is not supported")

        if not context_manager or context_manager.A_reduce is None:
            # not reduce dimension
            expand = lambda x: x
            f = lambda x: -self.acquisition.acquisition_function(x)[0,0]
            uniform_x = lambda : samples_multidimensional_uniform(self.acquisition.space.get_bounds(), 1)[0,:]
            dimension = self.acquisition.space.dimensionality
            print("not reduce: {} D".format(dimension))
        else:
            # reduce dimension
            expand = lambda x: context_manager._expand_vector(x)
            f = lambda x: -self.acquisition.acquisition_function(context_manager._expand_vector(x))[0,0]
            uniform_x = lambda : samples_multidimensional_uniform(context_manager.reduced_bounds, 1)[0,:]
            dimension = context_manager.space_reduced.dimensionality
            print("do reduce: {} D".format(dimension))

        # first sample
        s0 = uniform_x()

        res = scipy.optimize.basinhopping(f, x0=s0, niter=100)
        acq_min = res.fun
        #print("acq_min:",acq_min)

        # Now sample from x ~ p(x) = max(f(x) - acq_min, 0)
        # using No-U-Turn Sampler
        logp = lambda x: np.log(np.clip(f(x) - acq_min), a_min=0, a_max=None)
        start = smp.find_MAP(logp, {'x': s0})
        nuts = smp.NUTS(logp, start)
        chain = nuts.sample(self.N_sample, burn=self.warmup)

        # K-Means
        km = KMeans(n_clusters=self.batch_size)
        km.fit(chain.x)

        return expand(km.cluster_centers_)
