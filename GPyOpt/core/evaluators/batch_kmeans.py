# written by Shuhei Horiguchi

from .base import EvaluatorBase
from ...util.general import samples_multidimensional_uniform
import numpy as np
import scipy.optimize
from sklearn.cluster import KMeans

class KMBBO(EvaluatorBase):
    """
    Class for the batch method on 'Efficient and Scalable Batch Bayesian Optimization Using K-Means' (Groves et al., 2018).

    :param acquisition: acquisition function to be used to compute the batch.
    :param batch size: the number of elements in the batch.

    """
    def __init__(self, acquisition, batch_size, N_sample=200, N_rej=100):
        super(KMBBO, self).__init__(acquisition, batch_size)
        self.acquisition = acquisition
        self.batch_size = batch_size
        self.N_sample = N_sample
        self.N_rej = N_rej
        self.context_manager = acquisition.context_manager

    def compute_batch(self, duplicate_manager=None, context_manager=None, batch_context_manager=None):
        """
        Computes the elements of the batch.
        """

        assert not batch_context_manager or len(batch_context_manager) == self.batch_size
        if batch_context_manager:
            self.acquisition.optimizer.context_manager = batch_context_manager[0]
            raise NotImplementedError("batch_context is not supported")

        if self.context_manager.A_reduce is None:
            # not reduce dimension
            f = lambda x: -self.acquisition.acquisition_function(x)[0,0]
            uniform_x = lambda : samples_multidimensional_uniform(self.acquisition.space.get_bounds(), 1)[0,:]
            dimension = self.acquisition.space.dimensionality
        else:
            # reduce dimension
            f = lambda x: -self.acquisition.acquisition_function(self.context_manager._expand_vector(x))[0,0]
            uniform_x = lambda : samples_multidimensional_uniform(self.context_manager.reduced_bounds, 1)[0,:]
            dimension = self.context_manager.space_reduced.dimensionality

        # first sample
        s0 = uniform_x()

        res = scipy.optimize.basinhopping(f, x0=s0, niter=100)
        acq_min = res.fun
        #print("acq_min:",acq_min)
        uniform_u = lambda high: np.random.uniform(low=acq_min, high=high, size=1)[0]

        accepted_samples = np.empty((self.N_sample, dimension))
        accepted_samples[0] = s0

        # Batch Generarized Slice Sampling
        count=0
        #print("s0",s0)
        for i in range(1, self.N_sample):
            u = uniform_u(f(accepted_samples[i-1]))
            #print(i,"u:",u, "f(s):",f(accepted_samples[i-1]))
            while True:
                s = uniform_x()
                #print(s,f(s),u)
                if f(s) > u:
                    count = 0
                    #print(i,u,f(s), s)
                    accepted_samples[i] = s
                    break
                else:
                    count += 1
                    if count > self.N_rej:
                        count = 0
                        print("More than {} samples are rejected successively".format(self.N_rej))
                        accepted_samples[i] = s
                        break

        # K-Means
        km = KMeans(n_clusters=self.batch_size)
        km.fit(accepted_samples)

        return km.cluster_centers_
