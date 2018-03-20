import tensorflow as tf
import numpy as np

class Distribution:
    def __init__(self):
        pass
    def logdens(self, x):
        raise NotImplementedError
    def sample(self):
        raise NotImplementedError

class Normal(Distribution):
    def __init__(self, dim, mu=0, sigma=1):
        self.sigma = float(sigma)
        self.mu = float(mu)
        self.dim = dim

    def logdens(self, x, mean=False, full_reduce=True):
        s2 = tf.square(self.sigma)
        tmp = -tf.square(x-self.mu)/(2*s2) - 0.5 * tf.log(2*np.pi*s2)

        if full_reduce:
            idx = list(range(len(tmp.shape)))
        else:
            idx = [i for i in range(len(tmp.shape)) if i > 0]

        if mean:
            return tf.reduce_mean(tmp, idx)
        else:
            return tf.reduce_sum(tmp, idx)

    def sample(self):
        return tf.random_normal([self.dim], self.mu, self.sigma)

class NormalRW(Normal):
    def __init__(self, dim, mu=0, sigma=1, mu0=0, sigma0=1):
        super().__init__(dim, mu, sigma)
        self.init_distr = Normal(None, mu0, sigma0)

    def logdens(self, x, mean=False, full_reduce=True):
        assert len(x.shape) >= 2
        norms = x[:,1:] - x[:,:-1]
        ll = super().logdens(norms, mean, full_reduce) + self.init_distr.logdens(x[:,0:1], mean, full_reduce)
        return ll
    
    def sample(self):
        raise NotImplementedError
