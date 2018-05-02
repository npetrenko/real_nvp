import tensorflow as tf
from flows import NormalRW, DFlow, NVPFlow, LogNormal, GVAR, phase,Normal, floatX, MVNormal, MVNormalRW, Linear, LinearChol
from flows.models import VARmodel
import flows
from flows import parametrized as fp

import numpy as np
import pandas as pd
from tensorflow.contrib.distributions import WishartCholesky
import math
from tqdm import tqdm
import pickle as pkl

np.random.seed(1234)
tf.set_random_seed(1234)

ccodes = ['AUS', 'FRA', 'GBR']
datas = ['../../CDATA/{}.csv'.format(x) for x in ccodes]

datas = [pd.read_csv(x, index_col='VARIABLE').iloc[:,:-1] for x in datas]
columns = [data.columns.values.astype('float32') for data in datas]
datas = [data.values.astype(floatX) for data in datas]

scaler = 0.
max_year = 0

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

VAR_DIM = 4

YEARS = [x for x in data.columns if x > 2000]

country_data = {c:d for c,d in zip(ccodes, datas)}
#country_data = {c:d for c,d in zip(ccodes, datas) if c == 'FRA'}

NUM_SAMPLES=1024

#BUILDING the model

current_year = tf.placeholder(tf.float32, shape=(), name='current_year')
tf.summary.scalar('current_year', current_year)

for c, data in country_data.items():
    model = VARmodel(data, name='{}_model'.format(c), var_dim=4, current_year=current_year)
    for p in model.priors:
        tf.add_to_collection('priors', p)
    for l in model.logdensities:
        tf.add_to_collection('logdensities', l)

graph = tf.get_default_graph()

logdensity = tf.add_n(graph.get_collection('logdensities'))
print('logdensity: ', logdensity)

prior = tf.add_n(graph.get_collection('priors'))
print('prior: ', prior)

kl = logdensity - prior
print('KL: ', kl)
kl = tf.reduce_mean(kl)
kl /= 36*160


kls = tf.summary.scalar('KLd', kl)
summary = tf.summary.merge_all()

saver = tf.train.Saver()
with tf.variable_scope('build_upd') as upd_scope:
    vs = tf.trainable_variables()
    grads = tf.gradients(kl, vs)
    upd = zip(grads, vs)
    gnans = [tf.check_numerics(x, 'nan in {}'.format(x.op.name)) for x in grads if x is not None]
    with tf.control_dependencies(gnans):
        main_op = tf.train.AdamOptimizer(0.002).apply_gradients(upd)

sess = tf.InteractiveSession()
from flows.debug import wrapper
#sess = wrapper(sess)
#init = [tf.global_variables_initializer(), tf.variables_initializer(upd_vars)]
init = tf.global_variables_initializer()

sess.run(init)

writer = tf.summary.FileWriter('/tmp/tfdbg/frank_all')

def validate_year(year):
    cdic = {model.name:model for model in models}
    preds = {model.name:[] for model in models}
    preds_t = {model.name: model.preds for model in models}

    for step in range(3):
        preds_i = sess.run(preds_t, {current_year:year})
        preds_i = {k:v.mean(axis=1) for k,v in preds_i.items()}
        for k in preds.keys():
            preds[k].append(preds_i[k][cdic[k].years > year])
            
    mean_pred = {k:np.mean(v, axis=0) for k,v in preds.items()}
    for c, pred in mean_pred.items():
        pred_years = [x for x in YEARS if x > year]
        pred = pd.DataFrame(pred.T, columns=pred_years)
        mean_pred[c] = pred

    for model in models:
        try:
            a = model.data_raw.loc[:,year].values[:VAR_DIM]
        except KeyError:
            a = np.zeros(VAR_DIM, dtype=floatX)*np.nan
        mean_pred[model.name]['CYEAR={}'.format(year)] = a
    return mean_pred

#saver.restore(sess, './save/gvar_hier_fullcond1000-MP')

epoch = 0
for epoch in tqdm(range(500)):
    fd = {current_year:YEARS[0]}
    for step in range(10):
        sess.run(main_op, fd)
    s, _ = sess.run([summary, main_op], fd)
    writer.add_summary(s, global_step=epoch)
    if epoch % 30 == 0:
        saver.save(sess, './save/gvar_hier_fullcond1000-MP')

validations = []
for year in tqdm(YEARS):
    fd = {current_year: year}
    for epoch in range(epoch, epoch+10):
        for step in range(10):
            sess.run(main_op, fd)
        s, _ = sess.run([summary, main_op], fd)
        writer.add_summary(s, global_step=epoch)
    if year % 4 == 0:
        saver.save(sess, './save/gvar_hier_fullcond1000-MP')
    validations.append(validate_year(year))

    with open('output_gvar_hier_fullcond1000-MP.pkl', 'wb') as f:
        pkl.dump(validations,f)
