import tensorflow as tf
import numpy as np
from collections.abc import Sequence

class FlowSequence(Sequence):
    def __init__(self, flows = []):
        self.flows = flows
        super().__init__()
        
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
        
        if not inverse:
            for flow in self.flows:
                f = flow(f)
        else:
            for flow in self.flows[::-1]:
                f = flow(f, inverse=True)
                
        self.calc_logj()
        
        return f[-1].output
    
    def calc_logj(self):
        logjms = tf.stack([f.logjm for f in self.flows])
        logjm = tf.reduce_sum(logjms, axis=0)
        logj = tf.reduce_sum(logjm, axis=-1)
        self.logj = logj
        return logj

def Dense(inp, num_n, name='Dense', use_bias=True):
    with tf.variable_scope(name, initializer=tf.random_normal_initializer(mean=0, stddev=0.01)):
        inp_dim = int(inp.shape[-1])
        W = tf.get_variable('W', [inp_dim, num_n])
        pa = tf.matmul(inp, W)
        
        if use_bias:
            bb = 0
            if name=='preelastic':
                bb = -4
            b = tf.get_variable('b', [1, num_n]) + bb
            pa += b
            
    return pa

class NVPFlow:
    def __init__(self, dim=None, name='NVPFlow', output=None):
        self.dim = dim
        self.name = name
        self.output = output
        if output is not None:
            self.mask = np.zeros(dim, np.bool)
        
    def __call__(self, inp_flows=None, aux_vars=None, inverse=False):
        
        if isinstance(inp_flows, FlowSequence):
            prev_flow_output = inp_flows[-1].output
            dim = int(inp_flows[-1].dim)
            prev_flow = inp_flows[-1]
        elif isinstance(inp_flows, NVPFlow):
            prev_flow_output = inp_flows.output
            dim = inp_flows.dim
            inp_flows = FlowSequence([inp_flows])
            prev_flow = inp_flows[-1]
        else:
            raise ValueError('Input flow must be either a flowsequence or a flow')
            
        self.dim = dim
        if hasattr(prev_flow, 'aux_vars'):
            self.aux_vars = prev_flow.aux_vars
            if aux_vars is not None:
                raise ValueError('aux vars can be specified for single flow only')
        if aux_vars is not None:
            self.aux_vars = aux_vars
        
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
                prev_cover = np.zeros(dim, np.int)
                for flow in inp_flows:
                    prev_cover += flow.mask
                
                sort = np.argsort(prev_cover)[:dim//2]
                mask = np.zeros_like(prev_cover).astype('bool')
                mask[sort] = True
                #print(mask)

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
            rescaler = np.ones_like(mask).astype('float32')
            rescaler[np.logical_not(mask)] = 2
            #print(rescaler)
            
            mask = mask[np.newaxis,:]
            
            input_tensor = prev_flow_output*mask
            
            blend_tensor = prev_flow_output*(1 - mask)

            if hasattr(self, 'aux_vars'):
                blender = tf.concat([blend_tensor, self.aux_vars], axis=-1)
            else:
                blender = blend_tensor
            
            gate = Dense(blender, dim, name='preelastic')
            gate = tf.exp(gate)
            
            transition = Dense(blender, dim, name='transition')
            
            if not inverse:
                transformed = gate*input_tensor + transition
                self.output = transformed * mask + blend_tensor
                
                self.output += inp_flows[-1].output
                self.output /= rescaler
                
            else:
                restored = (input_tensor - transition)/(gate + 1)
                self.output = mask*restored + (1-mask)*blend_tensor
            
            self.logjm =  tf.log1p(gate*mask) - np.log(rescaler)
            
        return out_flows
