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

mean_std = 0.
for data in datas:
    std = np.std(data.values[:,1:] - data.values[:,:-1], axis=1)
    mean_std = std + mean_std
mean_std /= len(datas)
mean_std = np.concatenate([mean_std]*2, axis=0)
print('Mean std: {}'.format(mean_std))

max_year = 0
for i, data in enumerate(datas):
    data = data.astype(floatX)
    data.columns = data.columns.astype('float32')
    
    new_data = np.concatenate([data.values.T[1:], data.values.T[:-1]], axis=1)
    new_data_columns = data.columns[1:]
    new_data = pd.DataFrame(new_data.T/mean_std[:,np.newaxis], columns=new_data_columns)
    data = new_data
    datas[i] = data
    max_year = max(max(data.columns), max_year)

VAR_DIM = 4

YEARS = [3000.]#[x for x in data.columns if x > 2000]

country_data = {c:d for c,d in zip(ccodes, datas)}

NUM_SAMPLES=1024

#BUILDING the model

current_year = tf.placeholder(tf.float32, shape=(), name='current_year')
tf.summary.scalar('current_year', current_year)

#with tf.variable_scope('variation_rate', dtype=floatX):
    #variation_prior = tf.distributions.Exponential(rate=.3)
    #dim_ = (VAR_DIM*2+1)*VAR_DIM
    #variation_d = fp.pLogNormal(shape=[NUM_SAMPLES, dim_], mu=-3.)
    
    #variation = variation_d.sample()

    #pp = tf.cast(tf.reduce_sum(variation_prior.log_prob(tf.cast(variation, tf.float32)), axis=-1), floatX)
    #tf.add_to_collection('priors', pp)
#
    #tf.summary.histogram('variation', variation)
    #tf.summary.scalar('mean_variation', tf.reduce_mean(variation))

#with tf.variable_scope('global_inf'):
    #global_inf = DFlow([NVPFlow(dim=(VAR_DIM*2+1)*VAR_DIM, name='flow_{}'.format(i), aux_vars=variation) for i in range(6)], 
                        #init_sigma=0.01, num_samples=NUM_SAMPLES)

    #with tf.variable_scope('prior'):
        #pmat = np.ones([VAR_DIM, VAR_DIM*2+1], dtype=floatX)
        #pmat[:,:VAR_DIM] = 0.1
        #pmat[:,VAR_DIM:2*VAR_DIM] = 1.
        #pmat[:,-1] = 1.

        #global_sigma = tf.constant(pmat.reshape(-1), dtype=floatX)[tf.newaxis]
        #global_prior = Normal(None, sigma=global_sigma).logdens(global_inf.output, reduce=False)
        #global_prior = tf.reduce_sum(global_prior, axis=-1)
    #tf.add_to_collection('priors', global_prior)
    #tf.add_to_collection('logdensities', global_inf.logdens)

#print('Global output: ', global_inf.output)
#print('Global logdens: ', global_inf.logdens)

#individ_variation_prior = Normal(shape=None, sigma=variation, mu=global_inf.output, name='indiv_variation_prior')

models = []
indivs = {}

with tf.variable_scope(tf.get_variable_scope(), dtype=floatX, reuse=tf.AUTO_REUSE):
    for country, data in country_data.items():
        #with tf.variable_scope(country):
            #with tf.variable_scope('individ_variation'):
                #aux = tf.concat([global_inf.output, variation], axis=-1)
                #individ_variation = DFlow([NVPFlow((VAR_DIM*2+1)*VAR_DIM, 
                                                   #name='nvp_{}'.format(i), 
                                                   #aux_vars=aux) for i in range(6)], init_sigma=0.01, num_samples=NUM_SAMPLES)

                #ind = individ_variation.output + global_inf.output
            #indivs[country] = ind

            #tf.add_to_collection('logdensities', individ_variation.logdens)
            #tf.add_to_collection('priors', tf.reduce_sum(individ_variation_prior.logdens(ind, reduce=False), axis=-1))

        #raise mu=ind[tf.newaxis]
        model = VARmodel(data, name='{}_model'.format(country), var_dim=VAR_DIM, current_year=current_year, num_samples=NUM_SAMPLES)
        models.append(model)

graph = tf.get_default_graph()

fp = tf.reduce_sum(sum([model.priors for model in models], []), axis=0)
sp = 0.#tf.reduce_sum(graph.get_collection('priors'), axis=0)
prior = fp + sp

logdensity = tf.reduce_sum(sum([model.logdensities for model in models],[]), axis=0)# + tf.add_n(graph.get_collection('logdensities'))
print('logdensity ', logdensity)

kl = logdensity - prior
print('KL: ', kl)
kl = tf.reduce_mean(kl)
kl /= 36*200*4


kls = tf.summary.scalar('KLd', kl)
summary = tf.summary.merge_all()

saver = tf.train.Saver()
with tf.variable_scope('build_upd') as upd_scope:
    vs = tf.global_variables()
    grads = tf.gradients(kl, vs)
    upd = zip(grads, vs)
    gnans = [tf.check_numerics(x, 'nan in {}'.format(x.op.name)) for x in grads]
    with tf.control_dependencies(gnans):
        main_op = tf.train.AdamOptimizer(0.002).apply_gradients(upd)

#graph = tf.get_default_graph()
#upd_vars = graph.get_collection('variables', scope=upd_scope)

sess = tf.InteractiveSession()
from flows.debug import wrapper
#sess = wrapper(sess)
#init = [tf.global_variables_initializer(), tf.variables_initializer(upd_vars)]
init = tf.global_variables_initializer()

sess.run(init)

writer = tf.summary.FileWriter('/tmp/tfdbg/gvar_cond4_cont_corrected_prior_fix_disconnected3')

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
        s, _ = sess.run([summary, main_op], fd)
        writer.add_summary(s, global_step=epoch+470)
    if epoch % 30 == 0:
        saver.save(sess, './save/gvar_hier_fullcond1000-MP')

validations = []
for year in tqdm(YEARS):
    fd = {current_year: year}
    for epoch in range(epoch, epoch+10):
        for step in range(10):
            s, _ = sess.run([summary, main_op], fd)
            writer.add_summary(s, global_step=epoch)
    if year % 4 == 0:
        saver.save(sess, './save/gvar_hier_fullcond1000-MP')
    validations.append(validate_year(year))

    with open('output_gvar_hier_fullcond1000-MP.pkl', 'wb') as f:
        pkl.dump(validations,f)
