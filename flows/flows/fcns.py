import tensorflow as tf
import numpy as np
from .config import floatX

def Dense(inp, num_n, name='Dense', use_bias=True, activation=None):
    with tf.variable_scope(name, initializer=tf.random_normal_initializer(stddev=0.01, dtype=floatX)):
        inp_dim = int(inp.shape[-1])
        W = tf.get_variable('W', [inp_dim, num_n], dtype=floatX)
        pa = tf.matmul(inp, W)
        
        if use_bias:
            b = tf.get_variable('b', [1, num_n], dtype=floatX)
            pa += b
        if activation is not None:
            pa = activation(pa)
            
    return pa

def FCN(inp, num_n, num_hidden=60, num_layers=0, name='FCN'):
    with tf.variable_scope(name):
        layer = Dense(inp, num_hidden, name='init')
        layer = tf.nn.tanh(layer)
        for i in range(num_layers):
            layer = Dense(layer, num_hidden, name='hidden_' + str(i))
            layer = tf.nn.tanh(layer)
        layer = Dense(layer, num_n, name='final')
        return layer
