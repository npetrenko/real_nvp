import tensorflow as tf
import numpy as np
from .config import floatX
from tensorflow.python.ops.distributions.util import fill_triangular

class Distribution:
    def __init__(self, dim=None, name=None):
        self.dim = dim
        self.name = name
    def logdens(self, x):
        raise NotImplementedError
    def sample(self):
        raise NotImplementedError

class Normal(Distribution):
    def __init__(self, dim, mu=0, sigma=1):
        super().__init__(dim)
        self.sigma = np.array(sigma, dtype=floatX)
        self.mu = np.array(mu, dtype=floatX)

    def logdens(self, x, mean=False, full_reduce=True):
        s2 = tf.square(self.sigma)

        denum = - 0.5 * tf.log(2*np.pi*s2)
        tmp = -tf.square(x-self.mu)/(2*s2) + denum

        if full_reduce:
            idx = list(range(len(tmp.shape)))
        else:
            idx = [-1] if len(tmp.shape) != 0 else []

        if mean:
            return tf.reduce_mean(tmp, idx)
        else:
            return tf.reduce_sum(tmp, idx)

    def sample(self):
        return tf.random_normal(self.dim, self.mu, self.sigma, dtype=floatX)

class NormalRW(Normal):
    def __init__(self, dim, mu=0, sigma=1, mu0=0, sigma0=1):
        super().__init__(dim, mu, sigma)
        self.init_distr = Normal(None, mu0, sigma0)

    def logdens(self, x, mean=False, full_reduce=True):
        assert len(x.shape) >= 2
        norms = x[:,1:] - x[:,:-1]
        ll = super().logdens(norms, mean, full_reduce) + self.init_distr.logdens(x[:,0], mean, full_reduce)
        return ll
    
    def sample(self):
        raise NotImplementedError

class MVNormal(Distribution):
    def __init__(self, dim, sigma=1, name='mvn'):
        super().__init__(dim, name)
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            #lowerd = tf.get_variable('lowerd', initializer=tf.zeros([self.dim*(self.dim - 1)//2], dtype=floatX))
            lowerd = tf.get_variable('lowerd', initializer=np.random.normal(size=[self.dim + self.dim*(self.dim - 1)//2]).astype(floatX))

            ldiag = tf.get_variable('ldiag', initializer=np.array([-np.log(sigma)]*self.dim, dtype=floatX))
            diag_mask = tf.constant(np.diag(np.ones(self.dim)), name='diag_mask', dtype=floatX)

            diag = tf.diag(tf.exp(ldiag))
            fsigma = fill_triangular(lowerd)*(1-diag_mask) + diag
            isigma = tf.matmul(fsigma, tf.transpose(fsigma))

            self.fsigma = fsigma
            self.logdet = -2*tf.reduce_sum(ldiag) 
            self.inverse_sigma = isigma
            self.fsigma = fsigma

    def logdens(self, x, mean=False, full_reduce=True):
        with tf.name_scope(self.name):
            norms = tf.reduce_sum(tf.square(tf.tensordot(x, self.fsigma, [[-1], [0]])), axis=-1)
            tmp = -0.5*norms - (self.dim/2)*np.log(2*np.pi) - 0.5*self.logdet
            
            if full_reduce:
                idx = list(range(len(tmp.shape)))
            else:
                idx = [-1] if len(tmp.shape) != 0 else []

            if mean:
                return tf.reduce_mean(tmp, idx)
            else:
                return tf.reduce_sum(tmp, idx)

class MVNormalRW(MVNormal):
    def __init__(self, dim, sigma=1, sigma0=1, name='mvn_rw'):
        super().__init__(dim=dim, sigma=sigma, name=name)
        with tf.variable_scope(name, tf.AUTO_REUSE):
            self.init_distr = MVNormal(dim, sigma0, name='init_distr')

    def logdens(self, x, mean=False, full_reduce=True):
        assert len(x.shape) >= 2
        with tf.name_scope(self.name):
            with tf.name_scope('logdens'):
                norms = x[:,1:] - x[:,:-1]
                ll = super().logdens(norms, mean, full_reduce) + self.init_distr.logdens(x[:,0], mean, full_reduce)
            return ll
    
    def sample(self):
        raise NotImplementedError
