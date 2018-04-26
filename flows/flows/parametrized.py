from .config import floatX
from .basedistr import Normal, LogNormal

class pLogNormal:
    def __init__(self, dim, sigma=1., mu=0., name='pLogNormal'):
        with tf.variable_scope(name, initializer=tf.random_normal_initializer(stddev=0.01), dtype=floatX):
            logsigma = tf.get_variable('logsigma', shape=[dim]) + sigma
            mu = tf.get_variable('mu', shape=[dim]) + mu
            self.distr = LogNormal(dim=dim, sigma = tf.exp(logsigma), mu=mu, name='reparam_dist')
        self.sampled = False

    def sample(self):
        if self.sampled:
            raise ValueError('One cannot just sample this thing twice!')
        sample = self.distr.sample()

        ld = self.distr.logdens(ld, reduce=True)
        tf.add_to_collection('logdensities', ld)
        self.sampled = True
        return sample

class pNormal:
    def __init__(self, dim, sigma=1., mu=0., name='pNormal'):
        with tf.variable_scope(name, initializer=tf.random_normal_initializer(stddev=0.01), dtype=floatX):
            logsigma = tf.get_variable('logsigma', shape=[dim]) + sigma
            mu = tf.get_variable('mu', shape=[dim]) + mu
            self.distr = Normal(dim=dim, sigma = tf.exp(logsigma), mu=mu, name='reparam_dist')
        self.sampled = False

    def sample(self):
        if self.sampled:
            raise ValueError('One cannot just sample this thing twice!')
        sample = self.distr.sample()

        ld = self.distr.logdens(ld, reduce=True)
        tf.add_to_collection('logdensities', ld)
        self.sampled = True
        return sample
