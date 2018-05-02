import tensorflow as tf
from tensorflow.python.client import device_lib
from flows import NormalRW, DFlow, NVPFlow, LogNormal, GVAR, phase,Normal, floatX, MVNormal, MVNormalRW, Linear, LinearChol
import flows

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from tensorflow.contrib.distributions import WishartCholesky
import math
from flows.models import VARmodel

np.random.seed(1234)
tf.set_random_seed(1234)


ccodes = ['FRA', 'AUS', 'GBR']
datas = ['./CDATA/{}.csv'.format(x) for x in ccodes]

datas = [pd.read_csv(x, index_col='VARIABLE').iloc[:,:-1] for x in datas]
columns = [data.columns.values.astype('float32') for data in datas]
datas = [data.values.astype(floatX) for data in datas]

max_year = 0

scaler = 0.
for i, data in enumerate(datas):
    stds = (data[:,1:] - data[:,:-1]).std(axis=1)
    print(stds)
    scaler = scaler + stds
    datas[i] = data
print('---')
scaler /= len(datas)
for i in range(len(datas)):
    datas[i] /= scaler[:,np.newaxis]
    data = datas[i]
    stds = (data[:,1:] - data[:,:-1]).std(axis=1)
    print(stds)
    data = np.concatenate([data[:,1:], data[:,:-1]], axis=0)
    data = pd.DataFrame(data, columns=columns[i][1:])
    max_year = max(max_year, max(columns[i][1:]))
    datas[i] = data


country_data = {c:d for c,d in zip(ccodes, datas) if c == 'FRA'}


global_inf = DFlow([LinearChol(4*(4*2+1), name='lc')], init_sigma=0.01, num_samples=1024)

global_prior = Normal(shape=None, sigma=1., name='global_prior').logdens(global_inf.output, reduce=False)
global_prior = tf.reduce_sum(global_prior, axis=-1)
tf.add_to_collection('logdensities', global_inf.logdens)
tf.add_to_collection('priors', global_prior)

for c, d in country_data.items():
    model = VARmodel(d, name='{}_model'.format(c), 
                      var_dim=4, current_year=2000., mu=global_inf.output[:,tf.newaxis])
    for l in model.logdensities:
        tf.add_to_collection('logdensities', l)
    for p in model.priors:
        tf.add_to_collection('priors', p)

prior = tf.add_n(tf.get_collection('priors'))
logdensity = tf.add_n(tf.get_collection('logdensities'))
print(prior, logdensity)

kl = logdensity - prior
kl = tf.reduce_mean(kl)
kl /= 36*160

lr = tf.constant(0.002)
def build_upd(lr):
    return tf.train.AdamOptimizer(lr).minimize(kl)
    #vs = tf.trainable_variables()
    #gs = tf.gradients(kl, vs)
    #checks = [tf.check_numerics(x, 'NAN found in {}'.format(x.op.name)) for x in gs if x is not None]
    #with tf.control_dependencies(checks):
        ##main_op = tf.train.AdamOptimizer(lr).apply_gradients(zip(gs,vs))
    #return main_op

main_op = build_upd(lr)


kls = tf.summary.scalar('KLd', kl)
summary = tf.summary.merge_all()


from flows.debug import wrapper
sess = tf.InteractiveSession()
sess = wrapper(sess)

init = tf.global_variables_initializer()


epoch = 0

saver = tf.train.Saver()
saver.restore(sess, '/tmp/debug_save')


for epoch in range(epoch, 100000):
    print(epoch)
    for step in range(10):
        sess.run(main_op)

