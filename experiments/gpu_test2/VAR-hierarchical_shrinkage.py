import tensorflow as tf
from flows import NormalRW, DFlow, NVPFlow, LogNormal, GVAR, phase,Normal, floatX, MVNormal, MVNormalRW, Linear, LinearChol
from flows.models import VARmodel
import flows

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


#BUILDING the model

current_year = tf.placeholder(tf.float32, shape=(), name='current_year')
tf.summary.scalar('current_year', current_year)

def create_variation():
    with tf.variable_scope('variation_rate', dtype=floatX):
        variation_prior = tf.distributions.Exponential(rate=4.5)
        dim_ = (VAR_DIM*2+1)*VAR_DIM
        variation_mu = tf.get_variable('mu', shape=[dim_], initializer=tf.constant_initializer(math.log(0.3)))
        variation_presigma = tf.get_variable('presigma', shape=[dim_], initializer=tf.constant_initializer(-3.))
        variation_d = LogNormal(dim=dim_, mu=variation_mu, sigma=tf.exp(variation_presigma))
        
        variation = variation_d.sample()

        pp = tf.cast(tf.reduce_sum(variation_prior.log_prob(tf.cast(variation, tf.float32))), floatX)
        ld = variation_d.logdens(variation)
        tf.add_to_collection('logdensities', ld)
        tf.add_to_collection('priors', pp)

        tf.summary.histogram('variation', variation)
        tf.summary.scalar('mean_variation', tf.reduce_mean(variation))
    return variation

global_inf = DFlow([NVPFlow(dim=(VAR_DIM*2+1)*VAR_DIM, name='flow_{}'.format(i)) for i in range(6)], init_sigma=0.01)
global_prior = Normal(None, sigma=5.).logdens(global_inf.output)
tf.add_to_collection('priors', global_prior)
tf.add_to_collection('logdensities', global_inf.logdens[0])


models = []
indiv_logdens = []
indiv_priors = []
indivs = {}

with tf.variable_scope(tf.get_variable_scope(), dtype=floatX, reuse=tf.AUTO_REUSE):
    for country, data in country_data.items():
        with tf.variable_scope(country):
            variation = create_variation()
            individ_variation_prior = Normal((VAR_DIM*2+1)*VAR_DIM, sigma=variation, mu=global_inf.output[0])

            aux = tf.concat([global_inf.output, variation[tf.newaxis]], axis=-1)
            individ_variation = DFlow([NVPFlow((VAR_DIM*2+1)*VAR_DIM, 
                                               name='nvp_{}'.format(i), 
                                               aux_vars=aux) for i in range(6)], init_sigma=0.01)

            ind = individ_variation.output[0] + global_inf.output[0]
            indivs[country] = ind

            indiv_logdens.append(individ_variation.logdens)
            indiv_priors.append(individ_variation_prior.logdens(ind))

        model = VARmodel(data, name='{}_model'.format(country), var_dim=VAR_DIM, mu=ind[tf.newaxis], current_year=current_year)
        models.append(model)

graph = tf.get_default_graph()

prior = tf.reduce_sum([model.priors for model in models])+ tf.reduce_sum(indiv_priors) + tf.reduce_sum(graph.get_collection('priors'))

logdensity = tf.reduce_sum([model.logdensities for model in models])+ tf.reduce_sum(indiv_logdens) + tf.reduce_sum(graph.get_collection('logdensities'))

kl = logdensity - prior
kl /= 36*200*4


kls = tf.summary.scalar('KLd', kl)
summary = tf.summary.merge_all()


main_op = tf.train.AdamOptimizer(0.0001).minimize(kl)

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()

init.run()

writer = tf.summary.FileWriter('/tmp/tfdbg/gpu2')

def validate_year(year):
    cdic = {model.name:model for model in models}
    preds = {model.name:[] for model in models}
    preds_t = {model.name: model.preds for model in models}

    for step in range(1500):
        preds_i = sess.run(preds_t, {current_year:year})
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

saver = tf.train.Saver()

for epoch in tqdm(range(1000)):
    fd = {current_year:YEARS[0]}
    for step in range(100):
        sess.run(main_op, fd)
    s, _ = sess.run([summary, main_op], fd)
    writer.add_summary(s, global_step=epoch)

validations = []
for year in tqdm(YEARS):
    fd = {current_year: year}
    for epoch in range(epoch, epoch+80//4):
        for step in range(100):
            sess.run(main_op, fd)
        s, _ = sess.run([summary, main_op], fd)
        writer.add_summary(s, global_step=epoch)
    validations.append(validate_year(year))

    with open('output_evar_fullcond_rate4.5_final.pkl', 'wb') as f:
        pkl.dump(validations,f)
