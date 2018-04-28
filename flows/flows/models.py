import tensorflow as tf
import numpy as np
import pandas as pd
from .basedistr import *
from .flows import *
from tensorflow.contrib.distributions import WishartCholesky

class VARmodel:
    def __init__(self, data, name='VARmodel', var_dim=None, mu=None, current_year=None):
        self.num_samples = 2000
        self.data_raw = data
        self.mu = mu
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

        pd = np.std(data.values[:,1:] - data.values[:,:-1], axis=-1).astype(floatX)[:self.var_dim]

        with tf.variable_scope(name, dtype=floatX) as scope:
            self.data = tf.get_variable(initializer=data.values.T[np.newaxis].astype(floatX),
                                    trainable=False, name='data')
            self.scope = scope

            self.create_rw_priors()
            self.outputs = self.create_walk_inference(mu=mu)
            self.create_observ_dispersion_inference(pd*0.5)
            self.create_likelihood(self.observable_mask, self.outputs)

    def create_summary(self, stype, name, tensor):
        s = stype(name, tensor)
        self.summaries.append(s)

    def create_rw_priors(self):
        dim = self.dim
        with tf.variable_scope('rw_priors'):
            s1 = 0.01/4
            cov_prior = Normal(dim=None, mu=0.5*math.log(s1), sigma=3.5, name='cov_prior')

            with tf.variable_scope('PWalk_inf'):
                with tf.variable_scope('flows'):
                    flow_conf = [NVPFlow(dim=self.dim[0]*self.dim[1], name='nvp_{}'.format(i)) for i in range(4)] + \
                        [LinearChol(dim=self.dim[0]*self.dim[1], name='lc')]
                    ldiag = DFlow(flow_conf, num_samples=self.num_samples)
                    ldiag.output += 0.5*math.log(s1)
                    ldiag.logdens -= tf.reduce_sum(ldiag.output, axis=-1)
                    print('ldiag logdens', ldiag.logdens)

                    self.logdensities.append(ldiag.logdens)
                
                if self.mu is None:
                    sigma0 = None
                else:
                    sigma0 = 3.
                    
                PWalk = MVNormalRW(dim=self.dim[0]*self.dim[1], 
                                   sigma0=3., 
                                   diag=tf.exp(ldiag.output), name='OrdWalk')
                self.priors.append(tf.reduce_sum(cov_prior.logdens(ldiag.output, reduce=False), axis=1))
                self.PWalk = PWalk
                tf.summary.scalar('s1_ord', tf.reduce_mean(tf.sqrt(PWalk.diag)))

    def create_walk_inference(self, mu=None):
        dim = self.dim
        gvar = GVAR(dim=dim[0]*dim[1], len=self.NUM_STEPS, name='coef_rw_inference', 
                    mu=mu, num_samples=self.num_samples)
        outputs = gvar.sample()

        self.logdensities.append(gvar.logdens)
        with tf.name_scope('PWald_prior'):
            pwld = self.PWalk.logdens(outputs)
            print('pwld', pwld)
            self.priors.append(tf.reduce_sum(pwld, axis=[1]))
        self.outputs = outputs

        return outputs

    def create_observ_dispersion_inference(self, prior_disp):
        print('Prior disp: {}'.format(prior_disp))
        prior_loc = -0.5*np.log(prior_disp).astype(floatX)

        with tf.variable_scope('obs_d_inf', reuse=tf.AUTO_REUSE):
            flow_conf = [NVPFlow(dim=self.var_dim, name='nvp_{}'.format(i)) for i in range(6)] + \
                [LinearChol(dim=self.var_dim, name='lc')]
            ldiag = DFlow(flow_conf, init_sigma=0.05, num_samples=self.num_samples)

            ldiag.output -= prior_loc
            ldiag.logdens -= tf.reduce_sum(ldiag.output, axis=-1)

        self.obs_d = tf.distributions.Normal(tf.constant(0., dtype=floatX), 
                                             scale=tf.exp(ldiag.output), name='obs_d_prior')
        
        with tf.name_scope('obsrv_prior'):
            pr = tf.reduce_sum(tf.distributions.Normal(loc=prior_loc[np.newaxis], 
                                                       scale=tf.constant(3., dtype=floatX)).log_prob(ldiag.output), 
                               axis=-1)
        tf.summary.scalar('mean_ods', tf.reduce_mean(tf.exp(ldiag.output)))
        self.logdensities.append(ldiag.logdens)
        self.priors.append(pr)

    def predict(self, observable_mask, outputs):
        dim = self.dim
        data = self.data
        out = tf.reshape(outputs, [self.num_samples, self.NUM_STEPS, dim[0], dim[1]])
        out = tf.transpose(out, [1,0,2,3])

        def step(prev, x):
            mask = x[0]
            prev_pred = tf.where(mask, x[1], prev)
            params = x[2]

            d0 = params[:,:,:dim[0]]
            d1 = params[:,:,dim[0]:2*dim[0]]

            pp1 = prev_pred[:,:dim[0]]
            pp0 = prev_pred[:,dim[0]:2*dim[0]]

            new_pred = tf.einsum('bij,bj->bi', d0, pp0) + tf.einsum('bij,bj->bi', d1, pp1)+ params[:,:,-1] + pp1
            obs_noise = self.obs_d.sample()
            new_pred = tf.where(mask, new_pred, new_pred + obs_noise)

            new_pred = tf.concat([new_pred, pp1], axis=1)
            return new_pred
        
        data = tf.transpose(tf.tile(data, [self.num_samples, 1, 1]), [1,0,2])
        ar = tf.scan(step, [observable_mask, data, out], 
                     initializer=tf.zeros([self.num_samples, 2*dim[0]], dtype=floatX))
        return ar

    def create_likelihood(self, observable_mask, outputs):
        dim = self.dim
        obs_d = self.obs_d

        preds = self.predict(observable_mask, outputs)
        self.preds = preds[:,:self.var_dim]
        
        with tf.name_scope('loglikelihood'):
            data = tf.transpose(self.data, [1,0,2])
            diffs = preds[:-1,:] - data[1:,:]
            diffs = diffs[:,:,:dim[0]]

            logl = tf.reduce_sum(obs_d.log_prob(diffs), [2])
            print('logl', logl)
            logl *= tf.cast(self.observable_mask[1:], floatX)[:,tf.newaxis]

            logl = tf.reduce_sum(logl, axis=0)
            self.priors.append(logl)


class STACmodel:
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

        pd = np.std(data.values[:,1:] - data.values[:,:-1], axis=-1).astype(floatX)[:self.var_dim]

        with tf.variable_scope(name) as scope:
            self.data = tf.get_variable(initializer=data.values.T[np.newaxis],
                                    trainable=False, name='data')
            self.scope = scope

            self.outputs = self.create_walk_inference(mu=mu)
            self.create_observ_dispersion_inference(pd*0.5)
            self.create_likelihood(self.observable_mask, self.outputs)
            self.summary = tf.summary.merge(self.summaries)

    def create_summary(self, stype, name, tensor):
        s = stype(name, tensor)
        self.summaries.append(s)

    def create_walk_inference(self, mu=None):
        dim = self.dim

        if mu is not None:
            self.outputs = mu
        else:
            param_flow = DFlow([LinearChol(dim[0]*dim[1], name='lc')], init_sigma=0.01)
            outputs = param_flow.output

            self.outputs = outputs
            prior = Normal(None, mu=0., sigma=3.)
            self.priors.append(prior.logdens(self.outputs, reduce=True))
            self.logdensities.append(tf.reduce_sum(param_flow.logdens))

        return outputs

    def create_observ_dispersion_inference(self, prior_disp):
        print('Prior disp: {}'.format(prior_disp))
        with tf.variable_scope('obs_d_inf', reuse=tf.AUTO_REUSE):
            flow_conf = [NVPFlow(dim=self.var_dim, name='nvp_{}'.format(i)) for i in range(6)] + \
                [LinearChol(dim=self.var_dim, name='lc')]
            ldiag = DFlow(flow_conf, init_sigma=0.05)

            ldiag.output -= 0.5*np.log(prior_disp).astype(floatX)
            ldiag.logdens -= tf.reduce_sum(ldiag.output, axis=-1)

        self.obs_d = MVNormal(self.var_dim, sigma=None, name='obs_d_prior',
                   ldiag=ldiag.output[0])

        df = self.var_dim
        pmat = np.diag(2./prior_disp)/df
        cov_prior = WishartCholesky(df, pmat, cholesky_input_output_matrices=True)

        pr = cov_prior.log_prob(self.obs_d.fsigma)
        self.logdensities.append(ldiag.logdens[0])
        self.priors.append(pr)

        sigmas = tf.diag_part(self.obs_d.sigma)

        current_data = self.data[:,:self.OBSERV_STEPS]
        std = tf.nn.moments(current_data[0,1:,:self.var_dim] - current_data[0,:-1,:self.var_dim], axes=[0])[1]
        print(std, sigmas)
        rsquareds = 1 - sigmas/std
        self.create_summary(tf.summary.scalar, 'rsquared_post_mean', tf.reduce_mean(rsquareds))
        self.create_summary(tf.summary.histogram, 'post_rsquared', rsquareds)
        self.create_summary(tf.summary.histogram, 'post_disp', sigmas)

    def predict(self, observable_mask, outputs):
        dim = self.dim
        data = self.data
        out = tf.reshape(outputs, [dim[0], dim[1]])

        def step(prev, x):
            mask = x[0]
            prev_pred = tf.where(mask, x[1], prev)[tf.newaxis]
            params = out

            d0 = params[:,:dim[0]]
            d1 = params[:,dim[0]:2*dim[0]]

            pp1 = prev_pred[:,:dim[0]]
            pp0 = prev_pred[:,dim[0]:2*dim[0]]

            new_pred = tf.matmul(pp0, d0)[0] + tf.matmul(pp1, d1)[0]+ params[:,-1] + pp1[0]
            obs_noise = self.obs_d.sample()
            new_pred = tf.where(mask, new_pred, new_pred + obs_noise)

            new_pred = tf.concat([new_pred, pp1[0]], axis=0)
            return new_pred

        ar = tf.scan(step, [observable_mask, data[0]], initializer=tf.zeros([2*dim[0]], dtype=floatX))
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
