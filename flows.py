import tensorflow as tf
import numpy as np
from collections.abc import Sequence

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

class NVPFlow:
    def __init__(self, dim=None, name='NVPFlow', output=None):
        self.dim = dim
        self.name = name
        self.output = output
        if output is not None:
            self.mask = np.zeros(dim, np.bool)
        
    def __call__(self, inp_flows=None, inverse=False):
        
        if isinstance(inp_flows, FlowSequence):
            prev_flow_output = inp_flows[-1].output
            dim = int(inp_flows[-1].dim)
        elif isinstance(inp_flows, NVPFlow):
            prev_flow_output = inp_flows.output
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
                
            out_flows = FlowSequence([self])
        else:
            if hasattr(self, 'mask'):
                mask = self.mask
                
            else:
                prev_cover = np.zeros(dim, np.bool)
                for flow in inp_flows:
                    prev_cover += flow.mask
                uncovered = np.logical_not(prev_cover)
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
                
                self.mask = mask
            
            out_flows = inp_flows.add(self)
                
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