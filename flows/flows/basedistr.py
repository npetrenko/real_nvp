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
    def __init__(self, dim, sigma=1, name='mvn', lowerd=None, ldiag=None):
        super().__init__(dim, name)
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            if lowerd is None:
                lowerd = tf.get_variable('lowerd', initializer=tf.random_normal(shape=[self.dim + self.dim*(self.dim - 1)//2], dtype=floatX, stddev=0.05))
            else:
                print('Warning! Triengular part of lowerd is dropped and will cause problems with entropy calculations')

            if ldiag is None:
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
    def __init__(self, dim, sigma=1, sigma0=1, name='mvn_rw', lowerd=None, ldiag=None):
        super().__init__(dim=dim, sigma=sigma, name=name, lowerd=lowerd, ldiag=ldiag)
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

class DistLSTM:
    def __init__(self, dim, name='DistLSTM', sample_len=None, state_dim=64, num_layers=3, reuse=None):
        self.dim = dim
        self.name = name
        self.reuse = reuse
        self.sample_len = sample_len

        if sample_len is None:
            raise ValueError
        
        with tf.variable_scope(self.name, reuse=reuse) as scope:
            cells = [tf.nn.rnn_cell.LSTMCell(state_dim, 
                                                name='cell_{}'.format(i), 
                                                activation=tf.nn.tanh) for i in range(num_layers)]
            self.cell = tf.nn.rnn_cell.MultiRNNCell(cells)
            self.post_cell = lambda x: self.dense(x, dim, name='d1')
            self.init_dist = tf.get_variable('init_dist', [1,dim], initializer=tf.random_normal_initializer(stddev=0.01, mean=0.2))
            
            self.scope = scope
        
    def dense(self, inp, dim, name='dense'):
        with tf.variable_scope(name, initializer=tf.random_normal_initializer(stddev=0.01), reuse=self.reuse):
            W = tf.get_variable('W', [inp.shape[-1], dim])
            b = tf.get_variable('b', [1, dim])
            out = tf.matmul(inp, W) + b
        return out
    
    def logdens(self, seq):
        with tf.variable_scope(self.scope, reuse=True):
            batch_size, s_len = tf.shape(seq)[0], tf.shape(seq)[1]

            cell = self.cell

            s_t = tf.transpose(seq, [1,0,2])
            init_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)

            init = (tf.zeros([batch_size, cell.state_size[0][0]]), init_state)
            out,_ = tf.scan(lambda prev, x: cell(x, prev[1]), s_t, initializer=init)
            out = tf.transpose(out, [1,0,2])
            
            out_dim = out.shape
            
            out = tf.reshape(out, [-1, out_dim[-1]])
            out = self.post_cell(out)
            out = tf.reshape(out, [batch_size, s_len, self.dim])
            
            preds = out[:,:-1]
            target = seq[:,1:]
                        
            init_logits = tf.tile(self.init_dist, [batch_size,1])
            init_nll = tf.nn.softmax_cross_entropy_with_logits_v2(labels=seq[:,0], logits=init_logits)
            init_nll = init_nll[:,tf.newaxis]
            
            nll = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target, logits=preds)
            nll = tf.concat([init_nll, nll], axis=1)
            return -nll
        
    def sample(self):
        with tf.variable_scope(self.scope, reuse=True):
            init_sample = tf.distributions.Multinomial(total_count=1., logits=self.init_dist).sample()
            
            cell = self.cell

            init_state = cell.zero_state(batch_size=1, dtype=tf.float32)

            init = (init_sample, init_state)
            
            def step(prev):
                x = prev[0]
                state = prev[1]
                cell_step = cell(x, state)
                post_step = self.post_cell(cell_step[0])
                post_step = tf.distributions.Multinomial(total_count=1., logits=post_step).sample()
                return post_step, cell_step[1]
            
            out,_ = tf.scan(lambda prev, _: step(prev), tf.range(40), initializer=init)
            out = tf.transpose(out, [1,0,2])
            out = tf.concat([init_sample[:,tf.newaxis,:], out], axis=1)
            return out                
