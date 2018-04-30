import tensorflow as tf
from .config import floatX
from .basedistr import Normal, LogNormal

class pLogNormal:
    def __init__(self, shape, sigma=1e-3, mu=0., name='pLogNormal'):
        with tf.variable_scope(name, initializer=tf.random_normal_initializer(stddev=0.01), dtype=floatX):
            logsigma = tf.get_variable('logsigma', shape=shape) + sigma
            mu = tf.get_variable('mu', shape=shape) + mu
            self.distr = LogNormal(shape=shape, sigma = tf.exp(logsigma), mu=mu, name='reparam_dist')
        self.sampled = False

    def sample(self):
        if self.sampled:
            raise ValueError('One cannot just sample this thing twice!')
        sample = self.distr.sample()

        assert len(self.distr.shape) == 2

        ld = tf.reduce_sum(self.distr.logdens(sample, reduce=False), axis=-1)
        tf.add_to_collection('logdensities', ld)
        self.sampled = True
        return sample

class pNormal:
    def __init__(self, shape, sigma=1e-3, mu=0., name='pNormal'):
        with tf.variable_scope(name, initializer=tf.random_normal_initializer(stddev=0.01), dtype=floatX):
            logsigma = tf.get_variable('logsigma', shape=shape) + sigma
            mu = tf.get_variable('mu', shape=shape) + mu
            self.distr = Normal(shape=shape, sigma = tf.exp(logsigma), mu=mu, name='reparam_dist')
        self.sampled = False

    def sample(self):
        if self.sampled:
            raise ValueError('One cannot just sample this thing twice!')
        sample = self.distr.sample()

        assert len(self.distr.shape) == 2

        ld = tf.reduce_sum(self.distr.logdens(ld, reduce=False), axis=-1)
        tf.add_to_collection('logdensities', ld)
        self.sampled = True
        return sample
