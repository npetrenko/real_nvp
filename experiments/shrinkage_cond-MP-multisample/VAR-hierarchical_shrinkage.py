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

YEARS = [x for x in data.columns if x > 2000]

country_data = {c:d for c,d in zip(ccodes, datas)}

NUM_SAMPLES=1024

#BUILDING the model

current_year = tf.placeholder(tf.float32, shape=(), name='current_year')
tf.summary.scalar('current_year', current_year)

with tf.variable_scope('variation_rate', dtype=floatX):
    variation_prior = tf.distributions.Exponential(rate=.3)
    dim_ = (VAR_DIM*2+1)*VAR_DIM
    variation_d = fp.pLogNormal(shape=[NUM_SAMPLES, dim_], mu=math.log(0.2), sigma=-3.)
    
    variation = variation_d.sample()

    pp = tf.cast(tf.reduce_sum(variation_prior.log_prob(tf.cast(variation, tf.float32)), axis=-1), floatX)
    tf.add_to_collection('priors', pp)

    #tf.summary.histogram('variation', variation)
    tf.summary.scalar('mean_variation', tf.reduce_mean(variation))

with tf.variable_scope('global_inf'):
    global_inf = DFlow([NVPFlow(dim=(VAR_DIM*2+1)*VAR_DIM, name='flow_{}'.format(i), aux_vars=tf.log(variation)) for i in range(8)], 
                        init_sigma=0.01, num_samples=NUM_SAMPLES)

    with tf.variable_scope('prior'):
        pmat = np.ones([VAR_DIM, VAR_DIM*2+1], dtype=floatX)
        pmat[:,:VAR_DIM] = 0.1
        pmat[:,VAR_DIM:2*VAR_DIM] = 1.
        pmat[:,-1] = 1.

        global_sigma = tf.constant(pmat.reshape(-1), dtype=floatX)[tf.newaxis]
        global_prior = Normal(None, sigma=global_sigma).logdens(global_inf.output, reduce=[-1])
    tf.add_to_collection('priors', global_prior)
    tf.add_to_collection('logdensities', global_inf.logdens)

print('Global output: ', global_inf.output)
print('Global logdens: ', global_inf.logdens)

individ_variation_prior = Normal(shape=None, sigma=variation, mu=global_inf.output, name='indiv_variation_prior')

models = []
indivs = {}

with tf.variable_scope(tf.get_variable_scope(), dtype=floatX, reuse=tf.AUTO_REUSE):
    for country, data in country_data.items():
        with tf.variable_scope(country):
            with tf.variable_scope('individ_variation'):
                aux = tf.concat([global_inf.output, tf.log(variation)], axis=-1)
                individ_variation = DFlow([NVPFlow((VAR_DIM*2+1)*VAR_DIM, 
                                                   name='nvp_{}'.format(i), 
                                                   aux_vars=aux) for i in range(8)], init_sigma=0.01, num_samples=NUM_SAMPLES)

                ind = individ_variation.output + global_inf.output
            indivs[country] = ind

            tf.add_to_collection('logdensities', individ_variation.logdens)
            tf.add_to_collection('priors', individ_variation_prior.logdens(ind, reduce=[-1]))
                
        model = VARmodel(data, name='{}_model'.format(country), mu=ind[:, tf.newaxis], var_dim=VAR_DIM, current_year=current_year, num_samples=NUM_SAMPLES)
        models.append(model)
        for p in model.priors:
            tf.add_to_collection('priors', p)
        for l in model.logdensities:
            tf.add_to_collection('logdensities', l)
        print('\n')

graph = tf.get_default_graph()

logdensity = tf.add_n(graph.get_collection('logdensities'))
print('logdensity: ', logdensity)

prior = tf.add_n(graph.get_collection('priors'))
print('prior: ', prior)

kl = logdensity - prior
print('KL: ', kl)
kl = tf.reduce_mean(kl)
kl /= 36*200*4


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
#main_op = tf.train.AdamOptimizer(0.002).minimize(kl)

sess = tf.InteractiveSession()
from flows.debug import wrapper
#sess = wrapper(sess)
#init = [tf.global_variables_initializer(), tf.variables_initializer(upd_vars)]
init = tf.global_variables_initializer()

sess.run(init)

writer = tf.summary.FileWriter('/home/nikita/tmp/tfdbg/contin_fast')

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

saver.restore(sess, './save/gvar_hier_fullcond1000-MP')
#print('restoring from failsave...')
#saver.restore(sess, './save/failsave')
#print('restored')

def _run_main_op_faultcatch(fd):
    final_attempt = True
    for attempt in range(5):
        try:
            klv, _ = sess.run([kl, main_op], fd)
            #print('KL: ', klv)
            final_attempt = False
            break
        except tf.errors.InvalidArgumentError as loop_exc:
            print('\nNan found in gradients, retrying...')
            err = loop_exc
    if final_attempt:
        raise err

def run_main_op_faultcatch(fd):
    try:
        _run_main_op_faultcatch(fd)
    except tf.errors.InvalidArgumentError:
        if needs_save:
            saver.save(sess, './save/failsave')
        print('Saver current parameters in failsave')
        raise

needs_save=True
epoch = 0
for epoch in tqdm(range(epoch, 30)):
    fd = {current_year:YEARS[0]}
    for step in range(20):
        run_main_op_faultcatch(fd)
    try:
        s, klv = sess.run([summary,kl], fd)
        print('KL: ', klv)
        writer.add_summary(s, global_step=epoch)
    except tf.errors.InvalidArgumentError:
        print('Nan found in summary, skipping')
        pass
    if (epoch % 50 == 0) and (epoch != 0):
        if needs_save:
            saver.save(sess, './save/gvar_hier_fullcond1000-MP')
        pass

validations = []
for year in tqdm(YEARS):
    fd = {current_year: year}
    for epoch in range(epoch, epoch+20):
        for step in range(20):
            run_main_op_faultcatch(fd)
        try:
            s = sess.run(summary, fd)
            writer.add_summary(s, global_step=epoch)
        except tf.errors.InvalidArgumentError:
            pass
    if year % 4 == 0:
        if needs_save:
            saver.save(sess, './save/gvar_hier_fullcond1000-MP')
        pass
    validations.append(validate_year(year))

    with open('output_gvar_hier_fullcond1000-MP.pkl', 'wb') as f:
        pkl.dump(validations,f)
