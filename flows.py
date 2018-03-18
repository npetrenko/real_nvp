import tensorflow as tf
import numpy as np
from collections.abc import Sequence

phase = tf.placeholder_with_default(True, shape=(), name='learning_phase')

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
        init = NVPFlow(int(inp.shape[-1]), name='input_flow', output=inp)
        
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
    
def Dense(inp, num_n, name='Dense', use_bias=True):
    with tf.variable_scope(name, initializer=tf.random_normal_initializer(stddev=0.01)):
        inp_dim = int(inp.shape[-1])
        W = tf.get_variable('W', [inp_dim, num_n])
        pa = tf.matmul(inp, W)
        
        if use_bias:
            b = tf.get_variable('b', [1, num_n])
            pa += b
            
    return pa

class Flow:
    def __init__(self, dim=None, name=None, output=None):
        self.dim = dim
        self.name = name
        self.output = output
        if output is not None:
            self.mask = np.zeros(dim, np.bool)

    @staticmethod
    def _calc_mask(inp_flows):
        dim = inp_flows[-1].mask.shape[0]

        prev_cover = np.zeros(dim, np.int)
        for flow in inp_flows:
            prev_cover += flow.mask

        least_covered = np.argsort(prev_cover)
        mask = np.zeros(dim, np.bool)

        for i in least_covered[:min(len(least_covered)//2 + 1, dim-1)]:
            mask[i] = True
        
        return mask

    def __call__(self, inp_flows=None, inverse=False):

        if isinstance(inp_flows, FlowSequence):
            dim = int(inp_flows[-1].dim)
        elif isinstance(inp_flows, NVPFlow):
            dim = inp_flows.dim
            inp_flows = FlowSequence([inp_flows])
        else:
            raise ValueError('Input flow must be either a flowsequence or a flow')

        self.dim = dim
        
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
                mask = self._calc_mask(inp_flows)
                self.mask = mask
        return inp_flows

class NVPFlow(Flow):
    def __init__(self, dim=None, name='NVPFlow', output=None):
        super().__init__(dim, name, output)

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
            
            gate = Dense(blend_tensor, dim, name='preelastic')
            transition = Dense(blend_tensor, dim, name='transition')
            
            if not inverse:
                transformed = tf.exp(gate)*input_tensor + transition
                self.output = transformed * mask + blend_tensor
                
            else:
                transformed = (input_tensor - transition)/tf.exp(gate)
                self.output = transformed * mask + blend_tensor
            
            self.logj =  tf.reduce_sum(gate*mask, axis=-1, name='logj')
            
        return out_flows
    
class ResFlow(Flow):
    def __init__(self, dim=None, name='ResFlow', output=None):
        super().__init__(dim,name,output)
        
    def __call__(self, inp_flows=None, inverse=False):
        inp_flows = super().__call__(inp_flows, inverse)
        mask, dim = self.mask, self.dim

        if inp_flows is None:
            out_flows = FlowSequence([self])
        else:
            out_flows = inp_flows.add(self)
        prev_flow_output = inp_flows[-1].output
                
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            rescaler = np.ones_like(mask).astype('float32')
            rescaler[np.logical_not(mask)] = 2
            #print(rescaler)
            
            mask = mask[np.newaxis,:]
            
            input_tensor = prev_flow_output*mask
            
            blend_tensor = prev_flow_output*(1 - mask)
            
            gate = Dense(blend_tensor, dim, name='preelastic')
            gate = tf.log1p(tf.exp(gate))
            
            transition = Dense(blend_tensor, dim, name='transition')
            
            if not inverse:
                transformed = gate*input_tensor + transition
                self.output = transformed * mask + blend_tensor
                
                self.output += inp_flows[-1].output
                self.output /= rescaler
                
            else:
                restored = (input_tensor - transition)/(gate + 1)
                self.output = mask*restored + (1-mask)*blend_tensor
            
            self.logj =  tf.reduce_sum(tf.log1p(gate*mask) - np.log(rescaler), axis=-1)
        return out_flows
    
class BNFlow(Flow):
    def __init__(self, dim=None, name='BNFlow', output=None):
        super().__init__(dim,name,output)
        self.gamma = 0.99
    
    def _update_stats(self, inp_tensor):
        mu = tf.get_variable('mu', shape=inp_tensor.shape[1:], trainable=False)
        sigma2 = tf.get_variable('sigma2', trainable=False, initializer=tf.ones(inp_tensor.shape[1:]))
        
        offset = tf.get_variable('offset', shape=mu.shape)
        scale = tf.get_variable('scale', initializer=tf.zeros(inp_tensor.shape[1:]))
        scale = tf.identity(tf.log1p(tf.exp(scale)), name='scale')
        
        mean = tf.reduce_mean(inp_tensor, axis=0)
        mean2 = tf.reduce_mean(inp_tensor**2, axis=0)
        
        if not hasattr(self, 'ops'):
            op1 = mu.assign(mu*self.gamma + mean*(1 - self.gamma))
            op2 = sigma2.assign(sigma2*self.gamma + (mean2 - mean**2)*(1-self.gamma))
            self.ops = [op1, op2]
        
        self.collected = [mu, sigma2]
        self.adjust = [offset, scale]
        self.stats = [mean, mean2-mean**2]
        
    def __call__(self, inp_flows=None, inverse=False):
        inp_flows = super().__call__(inp_flows, inverse) 
        dim = self.dim
        
        self.mask = np.zeros(dim, np.bool)

        out_flows = inp_flows.add(self)
        prev_flow_output = inp_flows[-1].output
                
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self._update_stats(prev_flow_output)
            
            mean = tf.where(phase, self.stats[0], self.collected[0])
            var = tf.where(phase, self.stats[1], self.collected[1])
            
            if not inverse:
                self.output = self.adjust[0] + self.adjust[1]*(prev_flow_output - mean) / tf.sqrt(var)
            else:
                self.output = (prev_flow_output - self.adjust[0])*tf.sqrt(var) / self.adjust[1] + mean
            
            self.logj =  tf.reduce_sum(tf.log(self.adjust[1]), axis=-1, name='logj')
            self.logj -= 0.5*tf.reduce_sum(tf.log(var), axis=-1, name='logj')
            
        return out_flows
