import tensorflow as tf
import numpy as np

class Distribution:
    def __init__(self):
        pass
    def logdens(self, x):
        raise NotImplementedError

class Normal(Distribution):
    def __init__(self, dim, sigma=1, mu=0):
        self.sigma = sigma
        self.mu = mu
        self.dim = dim
    def logdens(self, x):
        s2 = tf.square(self.sigma)
        return -tf.square(x-self.mu)/(2*s2) - 0.5 * tf.log(2*np.pi*s2)
