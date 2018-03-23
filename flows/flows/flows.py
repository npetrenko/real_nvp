import tensorflow as tf
import numpy as np
from collections.abc import Sequence
from .basedistr import *
from .config import floatX
import random

phase = tf.placeholder_with_default(True, shape=(), name='learning_phase')

def softbound(x):
    b = 14.
    a = -14.
    assert b > a
    return a + (b-a)*(tf.atan(x/(b-a)) + np.pi/2)/np.pi
    #return x

class FlowSequence(Sequence):
    def __init__(self, flows = []):
        self.flows = flows
        super().__init__()
        
        ops = []
        for flow in flows:
            if hasattr(flow, 'ops'):
                ops += flow.ops
        self.ops = ops
        
    def add(self, flow):
        flows = self.flows[:]
        flows.append(flow)
        return FlowSequence(flows)
    
    def __getitem__ (self, i):
        return self.flows[i]
    
    def __len__ (self):
        return len(self.flows)
    
    def apply(self, inp, inverse=False):
        init = Flow(int(inp.shape[-1]), name='init_flow')
        init.output = inp
        init.logj = 0
        
        f = init
        ops = []
        
        if not inverse:
            for flow in self.flows:
                f = flow(f)

        else:
            for flow in self.flows[::-1]:
                f = flow(f, inverse=True)
                    
        self.ops = ops
                
        logj = 0
        for flow in self.flows:
            logj += flow.logj
        
        self.logj = logj
        
        return f[-1].output
    
    def calc_logj(self):
        logj = 0
        for f in self.flows:
            logj += f.logj
        
        self.logj = logj
        return logj

class DFlow:
    def __init__(self, flows, init_sigma=1.):
        base = Normal([1, flows[-1].dim], sigma=init_sigma)

        fseq = FlowSequence(flows)

        if not isinstance(fseq[-1], CFlow):
            bsamp = base.sample()
            out = fseq.apply(bsamp)

            self.base = base
            self.output = out
            self.fseq = fseq
            self.logdens = base.logdens(bsamp) - fseq.calc_logj()
        else:
            fseq[-1]()
            self.base = None
            self.output = fseq[-1].output
            self.fseq = fseq
            self.logdens = 0
        
    
def Dense(inp, num_n, name='Dense', use_bias=True):
    with tf.variable_scope(name, initializer=tf.random_normal_initializer(stddev=0.01, dtype=floatX)):
        inp_dim = int(inp.shape[-1])
        W = tf.get_variable('W', [inp_dim, num_n], dtype=floatX)
        pa = tf.matmul(inp, W)
        
        if use_bias:
            b = tf.get_variable('b', [1, num_n], dtype=floatX)
            pa += b
            
    return pa

class CFlow:
    def __init__(self, dim, name=None):
        self.dim = dim
        self.name = name

    def __call__(self, inp_flows=None):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            vals = tf.get_variable('W', shape=[self.dim], initializer=tf.random_normal_initializer(stddev=0.01), dtype=floatX)
            self.output = vals
            self.logj = 0
        return FlowSequence([self])

class Flow:
    def __init__(self, dim=None, name=None, aux_vars=None):
        self.dim = dim
        self.name = name
        self.aux_vars = aux_vars

    def __call__(self, inp_flows=None):
        if isinstance(inp_flows, FlowSequence):
            dim = int(inp_flows[-1].dim)
            assert (self.dim is None) or (self.dim == dim)
            self.dim = dim
        elif isinstance(inp_flows, Flow):
            dim = inp_flows.dim
            inp_flows = FlowSequence([inp_flows])
            assert (self.dim is None) or (self.dim == dim)
            self.dim = dim
        elif isinstance(inp_flows, tf.Tensor) or isinstance(inp_flows, tf.Variable):
            assert self.dim is not None
            fl = Flow(self.dim, 'input_flow')
            fl.output = inp_flows
            fl.logj = 0
            inp_flows = FlowSequence([fl])
        else:
            raise ValueError('Input flow must be either a flowsequence, a flow or a tensor')

        if self.dim is None:
            raise ValueError
        return inp_flows

class MaskedFlow(Flow):
    def __init__(self, dim, name=None, aux_vars=None):
        super().__init__(dim, name, aux_vars)

    def _calc_mask(self, inp_flows):
        dim = self.dim

        prev_cover = np.zeros(dim, np.int)
        for flow in inp_flows:
            if isinstance(flow, MaskedFlow):
                prev_cover += flow.mask

        if random.random() < 0.5:
            least_covered = np.argsort(prev_cover)
            mask = np.zeros(dim, np.bool)
            for i in least_covered[:min(len(least_covered)//2 + 1, dim-1)]:
                mask[i] = True
        else:
            uncovered = np.logical_not(prev_cover.astype('bool'))
            mask = uncovered

            if np.sum(mask) >= dim//2:
                ix = np.arange(len(mask))[mask]
                new_ix = np.random.choice(ix, size=dim//2, replace=False)
                new_mask = np.zeros_like(mask)
                new_mask[new_ix] = True
                mask = new_mask

            elif np.sum(mask) < dim//2:
                ix = np.arange(len(mask))[np.logical_not(mask)]
                new_ix = np.random.choice(ix, size=dim//2 - np.sum(mask), replace=False)
                new_mask = np.zeros_like(mask)
                new_mask[new_ix] = True
                mask += new_mask
            
        return mask

    def __call__(self, inp_flows=None, inverse=False):
        inp_flows = super().__call__(inp_flows)

        if inp_flows is None:
            if hasattr(self, 'mask'):
                mask = self.mask
            else:
                mask = np.zeros(dim, np.bool)
                mask[:dim//2] = True
                self.mask = mask
        else:
            if hasattr(self, 'mask'):
                mask = self.mask
            else:
                self.mask = self._calc_mask(inp_flows)
        return inp_flows

class Linear(Flow):
    def __init__(self, dim=None, name='LinFlow'):
        super().__init__(dim, name)

    def __call__(self, inp_flows=None, inverse=False):
        inp_flows = super().__call__(inp_flows)
        dim = self.dim
        
        if inp_flows is None:
            out_flows = FlowSequence([self])
        else:
            out_flows = inp_flows.add(self)
        prev_flow_output = inp_flows[-1].output
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE, dtype=floatX):
            W = tf.get_variable('W', initializer=tf.zeros([1, self.dim], dtype=floatX))
            b = tf.get_variable('b', initializer=tf.zeros([1, self.dim], dtype=floatX))

            gate = tf.exp(W)
            if not inverse:
                self.output = gate*prev_flow_output + b
            else:
                self.output = (prev_flow_output - b)/gate
            self.logj = tf.reduce_sum(W, axis=-1, name='logj')

        return out_flows


class NVPFlow(MaskedFlow):
    def __init__(self, dim=None, name='NVPFlow', aux_vars=None):
        super().__init__(dim, name, aux_vars)

    def __call__(self, inp_flows=None, inverse=False):
        inp_flows = super().__call__(inp_flows, inverse)
        mask, dim = self.mask, self.dim

        if inp_flows is None:
            out_flows = FlowSequence([self])

        else:
            out_flows = inp_flows.add(self)

        prev_flow_output = inp_flows[-1].output
                
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            mask = mask[np.newaxis,:]
            
            input_tensor = prev_flow_output*mask
            
            blend_tensor = prev_flow_output*(1 - mask)

            if self.aux_vars is not None:
                blend_tensor_full = tf.concat([blend_tensor, self.aux_vars], axis=-1)
            else:
                blend_tensor_full = blend_tensor
            
            gate = Dense(blend_tensor_full, dim, name='preelastic')
            transition = Dense(blend_tensor_full, dim, name='transition')
            gate = softbound(gate)
            
            if not inverse:
                transformed = tf.exp(gate)*input_tensor + transition
                self.output = transformed * mask + blend_tensor
                
            else:
                transformed = (input_tensor - transition)/tf.exp(gate)
                self.output = transformed * mask + blend_tensor
            
            self.logj =  tf.reduce_sum(gate*mask, axis=-1, name='logj')
            
        return out_flows
    
class ResFlow(MaskedFlow):
    def __init__(self, dim=None, name='ResFlow', aux_vars=None):
        super().__init__(dim, name, aux_vars)
        
    def __call__(self, inp_flows=None, inverse=False):
        inp_flows = super().__call__(inp_flows, inverse)
        mask, dim = self.mask, self.dim

        if inp_flows is None:
            out_flows = FlowSequence([self])
        else:
            out_flows = inp_flows.add(self)
        prev_flow_output = inp_flows[-1].output
                
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            rescaler = np.ones_like(mask).astype(floatX)
            rescaler[np.logical_not(mask)] = 2
            
            mask = mask[np.newaxis,:]
            
            input_tensor = prev_flow_output*mask
            
            blend_tensor = prev_flow_output*(1 - mask)
            
            if self.aux_vars is not None:
                blend_tensor_full = tf.concat([blend_tensor, self.aux_vars], axis=-1)
            else:
                blend_tensor_full = blend_tensor
            
            gate = Dense(blend_tensor_full, dim, name='preelastic')
            gate = softbound(gate)
            gate = tf.exp(gate)
            
            transition = Dense(blend_tensor_full, dim, name='transition')
            
            if not inverse:
                transformed = gate*input_tensor + transition
                self.output = transformed * mask + blend_tensor * (1-mask)
                
                self.output += inp_flows[-1].output
                self.output /= 2
                
            else:
                restored = 2*(input_tensor - 0.5*transition)/(gate + 1)
                self.output = mask*restored + (1-mask)*blend_tensor
            
            self.logj =  tf.reduce_sum(tf.log1p(gate*mask) - np.log(2), axis=-1)
        return out_flows
    
class BNFlow(Flow):
    def __init__(self, dim=None, name='BNFlow', output=None):
        super().__init__(dim,name)
        self.gamma = 0.99
    
    def _update_stats(self, inp_tensor):
        mu = tf.get_variable('mu', shape=inp_tensor.shape[1:], trainable=False, dtype=floatX)
        sigma2 = tf.get_variable('sigma2', trainable=False, initializer=tf.ones(inp_tensor.shape[1:], dtype=floatX))
        
        offset = tf.get_variable('offset', shape=mu.shape, dtype=floatX)
        scale = tf.get_variable('scale', initializer=tf.zeros(inp_tensor.shape[1:], dtype=floatX))
        scale = tf.identity(tf.log1p(tf.exp(scale)), name='scale')
        
        mean = tf.reduce_mean(inp_tensor, axis=0)
        mean2 = tf.reduce_mean(tf.square(inp_tensor), axis=0)
        
        if not hasattr(self, 'ops'):
            op1 = mu.assign(mu*self.gamma + mean*(1 - self.gamma))
            op2 = sigma2.assign(sigma2*self.gamma + (mean2 - mean**2)*(1-self.gamma))
            self.ops = [op1, op2]
        
        self.collected = [mu, sigma2]
        self.adjust = [offset, scale]
        self.stats = [mean, mean2-tf.square(mean)]
        
    def __call__(self, inp_flows=None, inverse=False):
        inp_flows = super().__call__(inp_flows) 
        dim = self.dim

        out_flows = inp_flows.add(self)
        prev_flow_output = inp_flows[-1].output
                
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE, dtype=floatX):
            with tf.variable_scope('update_stats', reuse=tf.AUTO_REUSE): 
                self._update_stats(prev_flow_output)
            
            mean = tf.where(phase, self.stats[0], self.collected[0])
            var = tf.where(phase, self.stats[1], self.collected[1])
            
            if not inverse:
                self.output = self.adjust[0] + self.adjust[1]*(prev_flow_output - mean) / tf.sqrt(var)
            else:
                self.output = (prev_flow_output - self.adjust[0])*tf.sqrt(var) / self.adjust[1] + mean
            
            with tf.name_scope('logj'):
                self.logj =  tf.reduce_sum(tf.log(self.adjust[1]), axis=-1, name='logj')
                self.logj -= 0.5*tf.reduce_sum(tf.log(var), axis=-1, name='logj_var')
            
        return out_flows
