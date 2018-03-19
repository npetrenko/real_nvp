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
    def __init__(self, dim, sigma=1, mu=0):
        self.sigma = float(sigma)
        self.mu = float(mu)
        self.dim = dim

    def logdens(self, x):
        s2 = tf.square(self.sigma)
        tmp = -tf.square(x-self.mu)/(2*s2) - 0.5 * tf.log(2*np.pi*s2)
        return tf.reduce_sum(tmp)

    def sample(self):
        return tf.random_normal([self.dim], self.mu, self.sigma)

class NormalRW(Normal):
    def __init__(self, dim, sigma=1, mu=0, mu0=0, sigma0=1):
        super().__init__(dim, sigma, mu)
        self.init_distr = Normal(None, mu0, sigma0)

    def logdens(self, x):
        assert len(x.shape) >= 2
        norms = x[:,1:] - x[:,:-1]
        ll = super().logdens(norms) + self.init_distr.logdens(x[:,0])
        return ll
    
    def sample(self):
        raise NotImplementedError
