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
datas = ['../CDATA/{}.csv'.format(x) for x in ccodes]

datas = [pd.read_csv(x, index_col='Unnamed: 0').iloc[:,:-1] for x in datas]

max_year = 0
for i, data in enumerate(datas):
    data.columns = data.columns.astype('int')
    data = data.astype(floatX)
    
    new_data = np.concatenate([data.values.T[1:], data.values.T[:-1]], axis=1)
    new_data_columns = data.columns[1:]
    new_data = pd.DataFrame(new_data.T, columns=new_data_columns)
    data = new_data
    datas[i] = data
    print(data.columns)
    max_year = max(max(data.columns), max_year)

YEARS = range(2000, max_year)

country_data = {c:d for c,d in zip(ccodes, datas)}


#BUILDING the model

current_year = tf.placeholder(tf.int32, shape=(), name='current_year')
tf.summary.scalar('current_year', current_year)


global_inf = DFlow([NVPFlow(dim=(3*2+1)*3, name='flow_{}'.format(i)) for i in range(6)], init_sigma=0.08)
global_prior = Normal(None, sigma=.3).logdens(global_inf.output)
tf.add_to_collection('priors', global_prior)
tf.add_to_collection('logdensities', global_inf.logdens)



with tf.variable_scope('variation_rate', dtype=floatX):
    variation_prerate = tf.get_variable('prerate',trainable=False, initializer=math.log(0.3))
    variation_rate = tf.exp(variation_prerate)
    variation = variation_rate#variation_d.sample()

    variation = tf.cast(variation, floatX)
    
    tf.summary.scalar('variation', variation)

individ_variation_prior = Normal((3*2+1)*3, sigma=variation, mu=global_inf.output[0])

models = []
indiv_logdens = []
indiv_priors = []
indivs = {}

with tf.variable_scope(tf.get_variable_scope(), dtype=floatX, reuse=tf.AUTO_REUSE):
    for country, data in country_data.items():
        with tf.variable_scope(country):
            individ_variation = DFlow([NVPFlow((3*2+1)*3, 
                                               name='nvp_{}'.format(i), 
                                               aux_vars=global_inf.output) for i in range(6)], init_sigma=0.08)

            ind = individ_variation.output[0]
            indivs[country] = ind

            indiv_logdens.append(individ_variation.logdens)
            indiv_priors.append(individ_variation_prior.logdens(ind))

        model = VARmodel(data, name='{}_model'.format(country), mu=ind[tf.newaxis], current_year=current_year)
        models.append(model)

indiv_logdens

graph = tf.get_default_graph()

prior = tf.reduce_sum([model.priors for model in models])+ tf.reduce_sum(indiv_priors) + tf.reduce_sum(graph.get_collection('priors'))

logdensity = tf.reduce_sum([model.logdensities for model in models])+ tf.reduce_sum(indiv_logdens) + tf.reduce_sum(graph.get_collection('logdensities'))

kl = logdensity - prior
kl /= 36*21


kls = tf.summary.scalar('KLd', kl)
summary = tf.summary.merge_all()


main_op = tf.train.AdamOptimizer(0.0001).minimize(kl)

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()

init.run()

writer = tf.summary.FileWriter('/home/nikita/tmp/hier/4main_rolling_long_rr1_restr')

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
        pred_years = list(range(year+1, year+len(pred)+1))
        pred = pd.DataFrame(pred.T, columns=pred_years)
        mean_pred[c] = pred

    for model in models:
        try:
            a = model.data_raw.loc[:,year].values[:3]
        except KeyError:
            a = np.zeros(3, dtype=floatX)*np.nan
        mean_pred[model.name]['CYEAR={}'.format(year)] = a
    return mean_pred

for epoch in tqdm(range(1500)):
    fd = {current_year:YEARS[0]}
    for step in range(100):
        sess.run(main_op, fd)
    s, _ = sess.run([summary, main_op], fd)
    writer.add_summary(s, global_step=epoch)

validations = []
for year in tqdm(YEARS):
    fd = {current_year: year}
    for epoch in range(epoch, epoch+150):
        for step in range(100):
            sess.run(main_op, fd)
        s, _ = sess.run([summary, main_op], fd)
        writer.add_summary(s, global_step=epoch)
    validations.append(validate_year(year))

    with open('output.pkl', 'wb') as f:
        pkl.dump(validations,f)
