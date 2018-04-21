import tensorflow as tf
import numpy as np
from .config import floatX
from tensorflow.python.ops.distributions.util import fill_triangular
import math

class Distribution:
    def __init__(self, dim=None, name=None):
        self.dim = dim
        self.name = name
        with tf.variable_scope(name) as scope:
            self.scope = scope
    def logdens(self, x):
        raise NotImplementedError
    def sample(self):
        raise NotImplementedError

class Normal(Distribution):
    def __init__(self, dim, mu=0., sigma=1., name='Normal'):
        super().__init__(dim, name)
        self.sigma = sigma
        self.mu = mu

    def logdens(self, x, reduce=True):
        with tf.variable_scope(self.scope):
            s2 = tf.cast(tf.square(self.sigma), floatX)

            denum = tf.cast(- 0.5 * tf.log(2*np.pi*s2), floatX)
            tmp = -tf.square(x-self.mu)/(2*s2) + denum
            if reduce:
                tmp = tf.reduce_sum(tmp)
            return tmp

    def sample(self):
        with tf.variable_scope(self.scope):
            return tf.random_normal(self.dim, self.mu, self.sigma, dtype=floatX)

class NormalRW(Normal):
    def __init__(self, dim, mu=0, sigma=1, mu0=0, sigma0=1, name='NormalRW'):
        super().__init__(dim, mu, sigma)
        Distribution.__init__(self, dim=dim, name=name)

        with tf.variable_scope(self.scope):
            self.init_distr = Normal(None, mu=mu0, sigma=sigma0)

    def logdens(self, x, reduce=True):
        with tf.variable_scope(self.scope):
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
                return dens
    
    def sample(self):
        raise NotImplementedError

class MVNormal(Distribution):
    def __init__(self, dim, sigma=1., name='MVNormal', lowerd=None, ldiag=None):
        super().__init__(dim=dim, name=name)
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
            self.fsigma = fsigma
            self.sigma = tf.linalg.inv(isigma)

    def logdens(self, x, reduce=True):
        with tf.variable_scope(self.scope):
            #reduction happens only on the last dimension:
            norms = tf.reduce_sum(tf.square(tf.tensordot(x, self.fsigma, [[-1], [0]])), axis=-1)
            tmp = -0.5*norms - (self.dim/2)*np.log(2*np.pi) - 0.5*(self.logdet + tf.cast(tf.shape(x)[-1], floatX)*tf.log(tf.cast(2*math.pi, floatX)))
            if reduce:
                return tf.reduce_sum(tmp)
            else:
                return tmp
    def sample(self):
        with tf.variable_scope(self.scope):
            # x^T.FS.FS^T.x => EPS = i(FS).i(FS.T)
            Ifsigma = tf.linalg.inv(self.fsigma)
            base = tf.random_normal([self.dim,1], dtype=floatX)
            return tf.matmul(Ifsigma, base)[:,0]

class MVNormalRW(MVNormal):
    def __init__(self, dim, sigma=1., sigma0=1., name='MVNormalRW', lowerd=None, ldiag=None):
        super().__init__(dim=dim, sigma=sigma, name=name, lowerd=lowerd, ldiag=ldiag)
        with tf.variable_scope(self.scope):
            self.init_distr = MVNormal(dim, sigma0, name='init_distr')

    def logdens(self, x, reduce=True):
        assert len(x.shape) >= 2
        with tf.variable_scope(self.scope):
            with tf.name_scope('logdens'):
                norms = x[:,1:] - x[:,:-1]
                if reduce:
                    ll = super().logdens(norms, reduce=reduce) + self.init_distr.logdens(x[:,0], reduce=reduce)
                else:
                    init_ll = self.init_distr.logdens(x[:,0], reduce=reduce)[:, tf.newaxis]
                    cont_ll = super().logdens(norms, reduce=reduce)
                    ll = tf.concat([init_ll, cont_ll], axis=1)
                    
            return ll
    
    def sample(self):
        raise NotImplementedError

class LogNormal(Distribution):
    def __init__(self, dim, mu=0., sigma=1., name='LogNormal'):
        super().__init__(dim, name)
        self.sigma = sigma
        self.mu = mu

    def logdens(self, x, reduce=True):
        with tf.variable_scope(self.scope):
            logits = tf.log(x)
            s2 = tf.cast(tf.square(self.sigma), floatX)

            denum = tf.cast(- 0.5 * tf.log(2*np.pi*s2), floatX)
            tmp = -tf.square(logits-self.mu)/(2.*s2) + denum - logits
            if reduce:
                tmp = tf.reduce_sum(tmp)
            return tmp

    def sample(self):
        with tf.variable_scope(self.scope):
            return tf.exp(tf.random_normal([self.dim], self.mu, self.sigma, dtype=floatX))

class DistLSTM:
    def __init__(self, dim, name='DistLSTM', sample_len=None, state_dim=64, num_layers=3, reuse=None, aux_vars=None):
        self.dim = dim
        self.name = name
        self.reuse = reuse
        self.sample_len = sample_len
        self.aux_vars = aux_vars

        if sample_len is None:
            raise ValueError
        
        with tf.variable_scope(self.name, reuse=reuse, dtype=floatX) as scope:
            cells = [tf.nn.rnn_cell.LSTMCell(state_dim, 
                                                name='cell_{}'.format(i), 
                                                activation=tf.nn.tanh) for i in range(num_layers)]
            self.cell = tf.nn.rnn_cell.MultiRNNCell(cells)
            self.post_cell = lambda x: self.dense(x, dim, name='d1')
            self.init_distr = tf.get_variable('init_distr', [1,dim], initializer=tf.random_normal_initializer(stddev=0.01, mean=0.2))
            
            self.scope = scope
        
    def dense(self, inp, dim, name='dense'):
        with tf.variable_scope(name, initializer=tf.random_normal_initializer(stddev=0.01), reuse=self.reuse, dtype=floatX):
            W = tf.get_variable('W', [inp.shape[-1], dim])
            b = tf.get_variable('b', [1, dim])
            out = tf.matmul(inp, W) + b
        return out
    
    def get_logdens(self, seq):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE, dtype=floatX):
            batch_size, s_len = tf.shape(seq)[0], tf.shape(seq)[1]
            print(batch_size, s_len)

            cell = self.cell

            s_t = tf.transpose(seq, [1,0,2])
            aux_vars = self.aux_vars

            if aux_vars is not None:
                aux_vars = tf.transpose(aux_vars, [1,0,2])

            init_state = cell.zero_state(batch_size=batch_size, dtype=floatX)

            init = (tf.zeros([batch_size, cell.state_size[0][0]], dtype=floatX), init_state)
            print(init[0])

            if aux_vars is None:
                out,_ = tf.scan(lambda prev, x: cell(x, prev[1]), s_t[:-1], initializer=init)
            else:
                out,_ = tf.scan(lambda prev, x: cell(tf.concat([x[0], x[1]], axis=-1), prev[1]), [s_t[:-1], aux_vars], initializer=init)
                
            out = tf.transpose(out, [1,0,2])
            
            out_dim = out.shape
            
            out = tf.reshape(out, [-1, out_dim[-1]])
            out = tf.cast(out, floatX)
            out = self.post_cell(out)
            out = tf.reshape(out, [batch_size, s_len-1, self.dim])
            
            preds = out
            target = seq[:,1:]
                        
            init_logits = tf.tile(self.init_distr, [batch_size,1])
            init_nll = tf.nn.softmax_cross_entropy_with_logits_v2(labels=seq[:,0], logits=init_logits)
            init_nll = init_nll[:,tf.newaxis]
            
            nll = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target, logits=preds)
            nll = tf.concat([init_nll, nll], axis=1)
            nll.set_shape(seq.shape[:2])
            return tf.identity(-nll, name='logdens')
        
    def sample(self):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE, dtype=floatX):
            init_sample_d = tf.distributions.Multinomial(total_count=1., logits=tf.cast(self.init_distr, tf.float32))
            init_sample = init_sample_d.sample()
            init_sample_logdens = tf.reduce_sum(tf.cast(init_sample_d.log_prob(init_sample), floatX))[tf.newaxis]
            init_sample = tf.cast(init_sample, floatX)

            aux_vars = self.aux_vars

            if aux_vars is not None:
                aux_vars = tf.transpose(aux_vars, [1,0,2])
            
            cell = self.cell

            init_state = cell.zero_state(batch_size=1, dtype=floatX)

            init = ((init_sample, tf.constant(0., dtype=floatX)), init_state)
            
            def step(prev):
                x = prev[0][0]
                print(x)
                state = prev[1]
                cell_step = cell(x, state)
                post_step = self.post_cell(cell_step[0])
                gen_distr = tf.distributions.Multinomial(total_count=1., logits=tf.cast(post_step, tf.float32))
                post_step = gen_distr.sample()
                post_step_logdens = tf.cast(tf.reduce_sum(gen_distr.log_prob(post_step)), floatX)
                print('Post_step', post_step)
                post_step = tf.cast(post_step, floatX)
                return (post_step, post_step_logdens), cell_step[1]
            
            if aux_vars is None:
                out,_ = tf.scan(lambda prev, _: step(prev), tf.range(self.sample_len-1), initializer=init)
            else:
                out,_ = tf.scan(lambda prev, x: step((tf.concat([prev[0], x], axis=-1), prev[1])), aux_vars, initializer=init)

            samples, ld = out
            samples = tf.transpose(samples, [1,0,2])
            samples = tf.concat([init_sample[:,tf.newaxis,:], samples], axis=1)

            ld = tf.concat([init_sample_logdens, ld], axis=0)[tf.newaxis]
            self.logdens = ld

            return tf.identity(samples, name='sample')

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
    def __init__(self, dim, len, name=None, mu=None):
        super().__init__(dim=dim, name=name)
        self.len = len
        self.logdens = None #property to hold logdensity of the last sample
        self.mu = mu

    def sample(self):
        from flows import LinearChol
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):

            init = Normal([self.len, self.dim], sigma=0.01)
            out_sample = init.sample() 

            with tf.variable_scope('VAR', dtype=floatX, initializer=tf.random_normal_initializer(stddev=0.001)):
                precholeskis = tf.get_variable('precholeskis', shape=[self.len,self.dim,self.dim])
                precholeskis_diag = tf.get_variable('precholeskis_diag', shape=[self.len,self.dim])

                aux_vars = tf.get_variable('aux_vars', initializer=tf.constant_initializer(0.), shape=[self.len, self.dim, self.dim])

                choleskis = precholeskis - tf.matrix_band_part(precholeskis, 0, -1)

                scan = True
                if scan:
                    def step(prev, x):
                        nv = x[0][tf.newaxis]
                        prev = prev[tf.newaxis]
                        chol = x[1]
                        aux_v = x[2]
                        return (tf.matmul(nv, chol) + tf.matmul(prev, aux_v))[0]
                        
                    outputs = tf.scan(step, elems=[out_sample, choleskis, aux_vars], initializer=tf.zeros(self.dim, dtype=floatX))
                else:
                    outputs = []
                    prev = tf.zeros(self.dim, dtype=floatX)
                    for i in range(self.len):
                        nv = out_sample[i][tf.newaxis]
                        chol = choleskis[i]
                        aux_v = aux_vars[i]
                        prev = prev[tf.newaxis]
                        prev = (tf.matmul(nv, chol) + tf.matmul(prev, aux_v))[0]
                        outputs.append(prev)
                    outputs = tf.stack(outputs)

                outputs += tf.exp(precholeskis_diag)*out_sample

                addmu = tf.get_variable('mu', initializer=tf.zeros([self.len,self.dim], dtype=floatX))
                addmu = tf.cumsum(addmu)
                outputs = tf.cumsum(outputs)

                if self.mu is not None:
                    addmu -= tf.reduce_mean(addmu, axis=0)[tf.newaxis]
                    addmu += self.mu

                outputs += addmu
                outputs = outputs[tf.newaxis]
                
                with tf.name_scope('logdens'):
                    self.logdens = -tf.reduce_sum(precholeskis_diag) + init.logdens(out_sample)
                    
                return outputs
