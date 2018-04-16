import tensorflow as tf
from .basedistr import Gumbel
from .fcns import Dense
import math

class DVAE:
    def __init__(self, dim, hshape, name='VAE', config=[128,64,32]):
        self.dim = dim
        self.hshape = hshape
        self.name = name
        self.config = config
        self.activation = tf.nn.tanh
        self.graph = tf.get_default_graph()
        with tf.variable_scope(self.name) as scope:
            self.create_controls()
            self.scope = scope

    def create_controls(self):
        with tf.variable_scope('controls') as control_scope:
            self.control_scope = control_scope
            pretemp = tf.get_variable('pretemp', shape=self.hshape, initializer=tf.constant_initializer(math.log(0.7)))
            self.temp = tf.exp(pretemp)
            tf.summary.histogram('temperature', self.temp)
            tf.summary.scalar('mean_temperature', tf.reduce_mean(self.temp))

            preeta = tf.get_variable('preeta', shape=(), initializer=tf.constant_initializer(0.))
            self.eta = tf.exp(preeta)
            tf.summary.scalar('eta', self.eta)

            with tf.variable_scope('RELAX') as relax_scope:
                self.relax_scope = relax_scope
            with tf.variable_scope('NVIL') as nvil_scope:
                self.nvil_scope = nvil_scope

    def build_relax(self, inp):
        with tf.variable_scope(self.relax_scope, reuse=tf.AUTO_REUSE):
            conf = [128, 64, 1]
            with tf.variable_scope('FCN'):
                for i, num_neurons in enumerate(conf):
                    if i != len(conf)-1:
                        activation = tf.nn.tanh
                    else:
                        activation = None
                    d = Dense(inp, num_neurons, name='d{}'.format(i), activation=activation)
                relax = d[:,0]
            return relax

    def build_nvil(self, inp):
        with tf.variable_scope(self.nvil_scope):
            conf = [128, 64, 1]
            with tf.variable_scope('FCN'):
                for i, num_neurons in enumerate(conf):
                    if i != len(conf)-1:
                        activation = tf.nn.tanh
                    else:
                        activation = None
                    d = Dense(inp, num_neurons, name='d{}'.format(i), activation=activation)
                nvil = d[:,0]
            return nvil

    def get_control_vars(self):
        control_vars = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.control_scope.name)
        return control_vars

    def get_encoder_vars(self):
        encoder_vars = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.encoder_scope.name)
        return encoder_vars

    def get_decoder_vars(self):
        decoder_vars = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.decoder_scope.name)
        return decoder_vars

    def build_control_loss(self, gradients):
        with tf.name_scope('control_loss'):
            gs = [tf.reduce_sum(tf.square(g)) for g in gradients]
            gs = tf.reduce_sum(gs)

            denum = [tf.reduce_prod(tf.shape(g)) for g in gradients]
            denum = tf.reduce_sum(denum)
            denum = tf.cast(denum, tf.float32)
            return gs/denum

    def priorkl(self, encoded, encoded_logits):
        with tf.name_scope('priorkl'):
            labels = tf.ones_like(encoded)/tf.cast(2, tf.float32)
            xent = tf.reduce_sum(encoded*tf.log(labels), axis=[-1,-2], name='xent')
            nent = tf.reduce_sum(encoded*tf.log(encoded), axis=[-1,-2], name='nent')
            with tf.control_dependencies([tf.check_numerics(x, message='priorkl_numerics_{}'.format(x.name)) for x in [xent, nent]]):
                loss = -xent + nent
        return loss

    def encode(self, x, hard=False, uniform_sample=None, condition=None):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE) as scope:
            self.encoder_scope = scope
            for i, out_dim in enumerate(self.config):
                x = Dense(x, out_dim, name='d{}'.format(i), activation=self.activation)

            with tf.variable_scope('latent_inf'):
                logits = Dense(x, self.hshape[0]*self.hshape[1], name='logits', activation=None)
                logits = tf.reshape(logits, [tf.shape(x)[0]] + self.hshape)
                self.logits = logits

                self.center_loss = 1e-2*tf.reduce_mean(tf.square(tf.log(tf.reduce_sum(tf.exp(logits), axis=-1))))

                gd = Gumbel(tf.shape(logits), logits=logits)
                encoded_gumb = gd.sample(us=uniform_sample, argmax=condition)
                self.uniform_sample = gd.uniform_sample

                encoded = tf.nn.softmax(encoded_gumb/self.temp)
                self.kl_loss = self.priorkl(tf.nn.softmax(self.logits), self.logits)
                encoded_soft = encoded

                encoded_hard = tf.distributions.Multinomial(1., logits=self.logits).sample()

                return encoded_soft, encoded_hard

    def decode(self, x):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE) as scope:
            self.decoder_scope = scope
            x = tf.reshape(x, [tf.shape(x)[0], self.hshape[0]*self.hshape[1]])
            for i, out_dim in enumerate(self.config[::-1]):
                x = Dense(x, out_dim, name='d{}'.format(i), activation=self.activation)
            x = Dense(x, self.dim, name='restoration', activation=None)
            return x
