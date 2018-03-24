from .config import floatX, phase
import tensorflow as tf
import numpy as np
    
class Normalizer:
    def __init__(self, name='normalizer'):
        self.name = name
        self.gamma = 0.9
        self.eps = 1e-4
    
    def _update_stats(self, inp_tensor):
        mu = tf.get_variable('mu', shape=inp_tensor.shape[1:], trainable=False, dtype=floatX)
        sigma2 = tf.get_variable('sigma2', trainable=False, initializer=tf.ones(inp_tensor.shape[1:], dtype=floatX))
        
        offset = tf.get_variable('offset', initializer=tf.zeros(inp_tensor.shape[1:], dtype=floatX))
        scale = tf.get_variable('scale', initializer=tf.zeros(inp_tensor.shape[1:], dtype=floatX))
        scale = tf.identity(tf.exp(scale), name='scale')
        
        mean = tf.reduce_mean(inp_tensor, axis=0)
        mean2 = tf.reduce_mean(tf.square(inp_tensor), axis=0)

        disp = mean2 - tf.square(mean)
        
        if not hasattr(self, 'ops'):
            op1 = mu.assign(mu*self.gamma + mean*(1 - self.gamma))
            op2 = sigma2.assign(sigma2*self.gamma + disp*(1-self.gamma))
            self.ops = [op1, op2]
        
        self.collected = [mu[tf.newaxis], sigma2[tf.newaxis]]
        self.adjust = [offset[tf.newaxis], scale[tf.newaxis]]
        self.stats = [mean[tf.newaxis], disp[tf.newaxis]]
        
    def __call__(self, inp_tensor, inverse=False):
                
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE, dtype=floatX):
            with tf.variable_scope('update_stats', reuse=tf.AUTO_REUSE, initializer=tf.random_normal_initializer(stddev=0.01, dtype=floatX)): 
                self._update_stats(inp_tensor)
            
            mean = tf.where(phase, self.stats[0], self.collected[0])
            var = tf.where(phase, self.stats[1], self.collected[1])
            
            if not inverse:
                output = self.adjust[0] + self.adjust[1]*(inp_tensor - mean) / tf.sqrt(var + self.eps)
            else:
                output = (inp_tensor - self.adjust[0])*tf.sqrt(var + self.eps) / self.adjust[1] + mean
            
        return output
