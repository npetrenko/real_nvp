import tensorflow as tf
import numpy as np
import pandas as pd
from .basedistr import *
from .flows import *
from tensorflow.contrib.distributions import WishartCholesky

class VARmodel:
    def __init__(self, data, name='VARmodel', var_dim=None, mu=None, current_year=None):
        self.data_raw = data
        self.var_dim = var_dim
        self.years = data.columns.values.astype('float32')
        years_c = tf.constant(data.columns.values, dtype=tf.float32, name='data_years')

        if current_year is None:
            self.OBSERV_STEPS = np.Infinity
        else:
            self.OBSERV_STEPS = tf.reduce_sum(tf.cast(years_c <= current_year, tf.int32))

        self.NUM_STEPS = data.shape[1]
        self.name = name
        self.logdensities = []
        self.priors = []
        self.dim = [self.var_dim,self.var_dim*2+1]
        self.summaries = []

        self.observable_mask = tf.range(0, self.NUM_STEPS, dtype=tf.int32) < self.OBSERV_STEPS

        pd = np.mean(np.std(data.values[:,1:] - data.values[:,:-1], axis=-1))

        with tf.variable_scope(name) as scope:
            self.data = tf.get_variable(initializer=data.values.T[np.newaxis],
                                    trainable=False, name='data')
            self.scope = scope

            self.create_rw_priors()
            self.outputs = self.create_walk_inference(mu=mu)
            self.create_observ_dispersion_inference(pd*0.5)
            self.create_likelihood(self.observable_mask, self.outputs)
            self.summary = tf.summary.merge(self.summaries)

    def create_summary(self, stype, name, tensor):
        s = stype(name, tensor)
        self.summaries.append(s)

    def create_rw_priors(self):
        dim = self.dim
        with tf.variable_scope('rw_priors'):
            with tf.variable_scope('walk_ord'):
                s1_prior_d = LogNormal(1, mu=math.log(0.01), sigma=6., name='s1_prior')

                with tf.variable_scope('s1_inference', dtype=floatX):
                    mu = tf.get_variable('mu', shape=[1],
                                         initializer=tf.constant_initializer(s1_prior_d.mu))

                    logsigma_init = tf.constant_initializer(min(math.log(s1_prior_d.sigma), -1.))
                    logsigma = tf.get_variable('logsigma', shape=[1],
                                               initializer=logsigma_init)
                    sigma = tf.exp(logsigma)
                    s1_d = LogNormal(1, mu=mu, sigma=sigma)

                s1 = s1_d.sample()

                self.create_summary(tf.summary.scalar, 's1_ord', s1[0])

                self.logdensities.append(s1_d.logdens(s1))

                s1_prior = s1_prior_d.logdens(s1)
                self.priors.append(s1_prior)

            PWalk = NormalRW(dim=None, sigma0=None, mu0=None, sigma=s1, name='OrdWalk')
            self.PWalk = PWalk

    def create_walk_inference(self, mu=None):
        dim = self.dim
        gvar = GVAR(dim=dim[0]*dim[1], len=self.NUM_STEPS, name='coef_rw_inference', mu=mu)
        outputs = gvar.sample()

        self.logdensities.append(gvar.logdens)
        self.priors.append(self.PWalk.logdens(outputs, reduce=True))
        self.outputs = outputs

        return outputs

    def create_observ_dispersion_inference(self, prior_disp):
        print('Prior disp: {}'.format(prior_disp))
        with tf.variable_scope('obs_d_inf', reuse=tf.AUTO_REUSE):
#             ldiag = DFlow([NVPFlow(dim=3, name='ldiag_flow_' + str(i)) for i in range(2)], init_sigma=0.05)
            ldiag = DFlow([LinearChol(dim=self.var_dim, name='ldiag_flow_' + str(i)) for i in range(1)], init_sigma=0.001)

            ldiag.output -= 0.5*math.log(prior_disp)
            ldiag.logdens -= tf.reduce_sum(ldiag.output, axis=-1)

        self.obs_d = MVNormal(self.var_dim, sigma=None, name='obs_d_prior',
                   ldiag=ldiag.output[0])

        df = self.var_dim
        pmat = np.diag([(2./prior_disp)]*self.var_dim)/df
        cov_prior = WishartCholesky(df, pmat, cholesky_input_output_matrices=True)

        pr = cov_prior.log_prob(self.obs_d.fsigma)
        self.logdensities.append(ldiag.logdens[0])
        self.priors.append(pr)

        sigmas = tf.sqrt(tf.diag_part(self.obs_d.sigma))

        std = tf.nn.moments(self.data[0,1:,:self.var_dim] - self.data[0,:-1,:self.var_dim], axes=[0])[1]
        print(std, sigmas)
        rsquareds = 1 - sigmas/std
        self.create_summary(tf.summary.scalar, 'rsquared_post_mean', tf.reduce_mean(rsquareds))
        self.create_summary(tf.summary.histogram, 'post_rsquared', rsquareds)
        self.create_summary(tf.summary.histogram, 'post_disp', sigmas)

    def predict(self, observable_mask, outputs):
        dim = self.dim
        data = self.data
        out = tf.reshape(outputs, [self.NUM_STEPS, dim[0], dim[1]])

        def step(prev, x):
            mask = x[0]
            prev_pred = tf.where(mask, x[1], prev)[tf.newaxis]
            params = x[2]

            d0 = params[:,:dim[0]]
            d1 = params[:,dim[0]:2*dim[0]]

            pp1 = prev_pred[:,:dim[0]]
            pp0 = prev_pred[:,dim[0]:2*dim[0]]

            new_pred = tf.matmul(pp0, d0)[0] + tf.matmul(pp1, d1)[0]+ params[:,-1] + pp1[0]
            obs_noise = self.obs_d.sample()
            new_pred = tf.where(mask, new_pred, new_pred + obs_noise)

            new_pred = tf.concat([new_pred, pp1[0]], axis=0)
            return new_pred

        ar = tf.scan(step, [observable_mask, data[0], out], initializer=tf.zeros([2*dim[0]], dtype=floatX))
        return ar

    def create_likelihood(self, observable_mask, outputs):
        dim = self.dim
        obs_d = self.obs_d

        preds = self.predict(observable_mask, outputs)
        self.preds = preds[:,:self.var_dim]

        diffs = preds[:-1] - self.data[0,1:]
        diffs = diffs[:,:dim[0]]

        std = tf.nn.moments(self.data[0,1:,:self.var_dim] - self.data[0,:-1,:self.var_dim], axes=[0])[1]
        od = tf.cast(self.observable_mask[1:], floatX)[:,tf.newaxis] * diffs
        rsq_obs = tf.reduce_mean(tf.square(od), axis=0)/std
        rsq_obs = 1-tf.reduce_mean(rsq_obs)
        self.create_summary(tf.summary.scalar, 'rsquared_observed', rsq_obs)

        logl = obs_d.logdens(diffs, reduce=False)
        logl *= tf.cast(self.observable_mask[1:], floatX)

        logl = tf.reduce_sum(logl)
        self.create_summary(tf.summary.scalar, 'loglikelihood', logl)
        self.priors.append(logl)
