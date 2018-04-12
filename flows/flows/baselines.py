import tensorflow as tf
from .config import floatX

class LSTMBL:
    def __init__(self, name='LSTMBL', state_dim=32, num_layers=3, reuse=None):
        self.name = name
        self.reuse = reuse
        with tf.variable_scope(self.name, reuse=reuse, dtype=floatX) as scope:
            cells = [tf.nn.rnn_cell.LSTMCell(state_dim, 
                                                name='cell_{}'.format(i), 
                                                activation=tf.nn.tanh) for i in range(num_layers)]
            self.cell = tf.nn.rnn_cell.MultiRNNCell(cells)
            self.post_cell = lambda x: self.dense(x, 1, name='d1')
            self.init_baseline = tf.get_variable('init_baseline', [1])
            self.scope = scope

    def dense(self, inp, dim, name='dense'):
        with tf.variable_scope(name, initializer=tf.random_normal_initializer(stddev=0.01), reuse=self.reuse, dtype=floatX):
            W = tf.get_variable('W', [inp.shape[-1], dim])
            b = tf.get_variable('b', [1, dim])
            out = tf.matmul(inp, W) + b
        return out

    def __call__(self, samples, samples_logprobs, target_f):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE, dtype=floatX):
            batch_size, s_len = tf.shape(samples)[0], tf.shape(samples)[1]

            cell = self.cell

            samples_t =  tf.transpose(samples, [1,0,2])
            slogp_t = tf.transpose(samples_logprobs, [1,0])

            init_state = cell.zero_state(batch_size=batch_size, dtype=floatX)

            init = ((tf.constant(0., dtype=floatX),tf.constant(0., dtype=floatX)), init_state)

            def step(x, prev):
                prev_state = prev[1]
                prev_sample = x[0]
                current_logp = x[1]
                print(prev_sample)
                current_pred_f, new_state = cell(prev_sample, prev_state)
                current_pred_f = self.post_cell(current_pred_f)[0,0]
                cell_loss = tf.reduce_sum(tf.square(current_pred_f - target_f))
                print(current_logp)
                reinforce_target = current_logp[0]*tf.stop_gradient(target_f - current_pred_f)
                print(current_pred_f)
                return (reinforce_target, cell_loss), new_state

            out,_ = tf.scan(lambda prev, x: step(x, prev), [samples_t[:-1], slogp_t[1:]], initializer=init)

            init_reinforce = slogp_t[0]*tf.stop_gradient(target_f - self.init_baseline)
            print(self.init_baseline)
            tmp = tf.divide(tf.square(self.init_baseline - target_f), tf.cast(s_len, floatX))
            print(tmp)
            loss = tf.cast(tf.reduce_mean(out[1]), floatX) + tmp
            self.loss = loss
            reinforce_loss = tf.cast(tf.reduce_sum(out[0]), floatX) + init_reinforce
                
            return reinforce_loss
    def get_grads(self):
        vs = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope.name)
        grads = tf.gradients(self.loss, vs)
        return [(g,v) for g,v in zip(grads, vs)]
