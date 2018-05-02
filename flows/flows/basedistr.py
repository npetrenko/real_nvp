import tensorflow as tf
import numpy as np
from .config import floatX
from tensorflow.python.ops.distributions.util import fill_triangular
import math

class Distribution:
    def __init__(self, shape=None, name=None):
        self.shape = shape
        self.name = name
        with tf.variable_scope(name) as scope:
            self.scope = scope
    def logdens(self, x):
        raise NotImplementedError
    def sample(self):
        raise NotImplementedError

class Normal(Distribution):
    def __init__(self, shape, mu=0., sigma=1., name='Normal'):
        super().__init__(shape=shape, name=name)
        self.sigma = sigma
        self.mu = mu

    def logdens(self, x, reduce=True):
        with tf.variable_scope(self.scope):
            with tf.name_scope('logdens'):
                if isinstance(x,tf.Tensor):
                    inp_dtype = x.dtype
                else:
                    inp_dtype = floatX

                print(inp_dtype)
                s2 = tf.cast(tf.square(self.sigma), inp_dtype)

                denum = tf.cast(- 0.5 * tf.log(2*np.pi*s2), inp_dtype)
                tmp = -tf.square(x-self.mu)/(2*s2) + denum
                if reduce:
                    tmp = tf.reduce_sum(tmp)
                tmp = tf.identity(tmp, name='logdens')
                return tmp

    def sample(self):
        with tf.variable_scope(self.scope):
            with tf.name_scope('sample'):
                return tf.random_normal(self.shape, self.mu, self.sigma, dtype=floatX)

class NormalRW(Normal):
    def __init__(self, dim, mu=0, sigma=1, mu0=0, sigma0=1, name='NormalRW'):
        super().__init__(shape=None, mu=mu, sigma=sigma)
        Distribution.__init__(self, shape=None, name=name)

        with tf.variable_scope(self.scope):
            self.init_distr = Normal(None, mu=mu0, sigma=sigma0)

    def logdens(self, x, reduce=True):
        with tf.variable_scope(self.scope):
            with tf.name_scope('logdens'):
                assert len(x.shape) >= 2
                norms = x[:,1:] - x[:,:-1]
                if reduce==True:
                    if self.init_distr.mu is None:
                        init_logdens = 0.
                        if self.init_distr.sigma is not None:
                            raise ValueError
                    else:
                        init_logdens = self.init_distr.logdens(x[:,0], reduce)
                        if self.init_distr.sigma is None:
                            raise ValueError
                    ll = super().logdens(norms, reduce) + init_logdens
                    ll = tf.identity(ll, name='logdens')
                    return ll
                else:
                    if self.init_distr.mu is not None:
                        print('Init_distr.mu is not None')
                        init_dens = tf.reduce_sum(self.init_distr.logdens(x[:,0], reduce)[:,tf.newaxis], axis=-1)
                    else:
                        init_dens = tf.constant([[0.]], dtype=floatX)
                        if self.init_distr.sigma is not None:
                            raise ValueError
                    cont_dens = tf.reduce_sum(super().logdens(norms, reduce), axis=-1)
                    dens = tf.concat([init_dens, cont_dens], axis=1)
                    dens = tf.identity(dens, 'logdens')
                    return dens

class MVNormal(Distribution):
    def __init__(self, dim, sigma=1., name='MVNormal', lowerd=None, ldiag=None):
        super().__init__(shape=[dim], name=name)
        self.dim = dim
        with tf.variable_scope(self.scope):
            if lowerd is None:
                lowerd = tf.get_variable('lowerd', initializer=tf.random_normal(shape=[self.dim + self.dim*(self.dim - 1)//2], dtype=floatX, stddev=0.05))
            else:
                print('Warning! Diagonal part of lowerd is dropped and will cause problems with entropy calculations')

            if ldiag is None:
                ldiag = tf.get_variable('ldiag', initializer=np.array([-np.log(sigma)]*self.dim, dtype=floatX))

            diag_mask = tf.constant(np.diag(np.ones(self.dim)), name='diag_mask', dtype=floatX)

            diag = tf.diag(tf.exp(ldiag))
            fsigma = fill_triangular(lowerd)*(1-diag_mask) + diag
            isigma = tf.matmul(fsigma, tf.transpose(fsigma))

            self.fsigma = fsigma
            self.logdet = -2*tf.reduce_sum(ldiag) 
            self.inverse_sigma = isigma
            self.sigma = tf.linalg.inv(isigma)

    def logdens(self, x, reduce=True):
        with tf.variable_scope(self.scope):
            with tf.name_scope('logdens'):
                #reduction happens only on the last dimension
                x_shape = tf.shape(x)
                x = tf.reshape(x, [-1, x.shape[-1]])
                resid_shape = x_shape[:-1]

                norms = tf.reduce_sum(tf.square(tf.matmul(x, self.fsigma)), axis=-1)
                print(norms)
                tmp = -0.5*norms - (self.dim/2)*np.log(2*np.pi) - 0.5*(self.logdet + tf.cast(tf.shape(x)[-1], floatX)*math.log(2*math.pi))
                if reduce:
                    return tf.reduce_sum(tmp, name='logdens')
                else:
                    return tf.reshape(tmp, resid_shape, name='logdens')
    def sample(self):
        with tf.variable_scope(self.scope):
            with tf.name_scope('sample'):
                # x^T.FS.FS^T.x => EPS = i(FS).i(FS.T)
                Ifsigma = tf.linalg.inv(self.fsigma)
                base = tf.random_normal([self.dim,1], dtype=floatX)
                return tf.identity(tf.matmul(Ifsigma, base)[:,0], name='sample')

class MVNormalRW(Distribution):
    def __init__(self, dim, sigma0=1., name='MVNormalRW', diag=None):
        super().__init__(name=name, shape=None)
        self.diag = diag
        with tf.variable_scope(self.scope):
            if sigma0 is not None:
                self.init_distr = Normal(dim, sigma=sigma0, name='init_distr')
            else:
                self.init_distr = Distribution(name='dummy_init_distr')
                self.init_distr.logdens = lambda *x, **y: tf.constant(0., dtype=floatX)

    def logdens(self, x):
        assert len(x.shape) >= 2
        with tf.variable_scope(self.scope):
            with tf.name_scope('logdens'):
                norms = x[:,1:] - x[:,:-1]

                init_ll = tf.reduce_sum(self.init_distr.logdens(x[:,0], reduce=False), axis=-1)
                cont_ll = tf.reduce_sum(Normal(shape=None, mu=tf.constant(0., dtype=floatX), 
                                               sigma=self.diag[:,tf.newaxis,:]).logdens(norms, reduce=False), axis=-1)
                ll = tf.concat([init_ll[:,tf.newaxis], cont_ll], axis=1, name='logdens')    
            return ll

class LogNormal(Distribution):
    def __init__(self, shape, mu=0., sigma=1., name='LogNormal'):
        super().__init__(shape=shape, name=name)
        self.sigma = sigma
        self.mu = mu

    def logdens(self, x, reduce=True):
        with tf.variable_scope(self.scope):
            with tf.name_scope('logdens'):
                logits = tf.log(x)
                s2 = tf.cast(tf.square(self.sigma), floatX)

                denum = tf.cast(- 0.5 * tf.log(2*np.pi*s2), floatX)
                tmp = -tf.square(logits-self.mu)/(2.*s2) + denum - logits
                if reduce:
                    tmp = tf.reduce_sum(tmp)
                tmp = tf.identity(tmp, name='logdens')
                return tmp

    def sample(self):
        with tf.variable_scope(self.scope):
            with tf.name_scope('sample'):
                return tf.identity(tf.exp(tf.random_normal(self.shape, self.mu, self.sigma, dtype=floatX)), name='sample')


class Gumbel:
    def __init__(self, shape, logits, name='Gumbel'):
        self.shape = shape
        self.name = name
        with tf.name_scope(self.name):
            with tf.name_scope('normalized_logits'):
                self.logits = logits - tf.log(tf.reduce_sum(tf.exp(logits), axis=-1))[...,tf.newaxis]
    def sample(self, us=None, argmax=None):
        with tf.name_scope(self.name):
            if us is None:
                us = tf.cast(tf.random_uniform(self.shape, minval=1e-3, maxval=1-1e-3), floatX)
            self.uniform_sample = us
            
            if argmax is None:
                gb = -tf.log(-tf.log(us)) + self.logits
            else:
                upper = -tf.log(-tf.log(us))
                argmax_ix = tf.argmax(argmax, axis=-1)
                upper_samples = tf.reduce_sum(us*argmax, axis=-1)
                gb = -tf.log(-tf.log(us)/tf.exp(self.logits) - tf.log(upper_samples)[...,tf.newaxis])
                gb = upper*argmax + gb*(1-argmax)
            return gb


class GVAR(Distribution):
    def __init__(self, dim, len, name=None, num_samples=1, centralize=False):
        super().__init__(shape=None, name=name)
        self.dim = dim
        self.len = len
        self.num_samples = num_samples
        self.logdens = None #property to hold logdensity of the last sample
        self.centralize = centralize

    def sample(self):
        from flows import LinearChol
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            with tf.name_scope('sample'):
                init = Normal([self.len, self.num_samples, self.dim], sigma=0.001)
                out_sample = init.sample() 

                with tf.variable_scope('VAR', dtype=floatX, initializer=tf.random_normal_initializer(stddev=0.001)):
                    precholeskis = tf.get_variable('precholeskis', shape=[self.len,self.dim,self.dim])
                    precholeskis_diag = tf.get_variable('precholeskis_diag', shape=[self.len,self.dim])

                    aux_vars = tf.get_variable('aux_vars', shape=[self.len, self.dim, self.dim])

                    choleskis = precholeskis - tf.matrix_band_part(precholeskis, 0, -1)

                    def step(prev, x):
                        nv = x[0]
                        prev = prev
                        chol = x[1]
                        aux_v = x[2]
                        return tf.matmul(nv, chol) + tf.matmul(prev, aux_v) 
                        
                    outputs = tf.scan(step, elems=[out_sample, choleskis, aux_vars], initializer=tf.zeros([self.num_samples, self.dim], dtype=floatX))

                    addmu = tf.get_variable('mu', shape=[self.len,1,self.dim])

                    outputs += tf.exp(precholeskis_diag)[:,tf.newaxis,:]*out_sample

                    outputs += addmu

                    outputs = tf.cumsum(outputs)

                    outputs = tf.transpose(outputs, [1,0,2])
                    
                    with tf.name_scope('logdens'):
                        self.logdens = -tf.reduce_sum(precholeskis_diag) + tf.reduce_sum(init.logdens(out_sample, reduce=False), [0,2])

                    outputs = tf.identity(outputs, name='sample')
                    return outputs
