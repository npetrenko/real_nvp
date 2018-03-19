
# coding: utf-8

# In[1]:


import tensorflow as tf
from flows import NormalRW, DFlow, NVPFlow, phase, Normal
import numpy as np
from matplotlib import pyplot as plt
from flows.debug import wrapper as swrap

np.random.seed(1234)


# In[2]:


n = 20
s1 = 0.1
m1 = 0.
dim = [3,4]

params = []
params.append(np.random.normal(size=dim))
for i in range(n-1):
    new = params[i] + np.random.normal(loc=m1, scale=s1, size=dim)
    params.append(new)
params = np.array(params)

PWalk = NormalRW(dim[0]*dim[1], sigma=s1, mu=m1)


# In[3]:


params.shape


# In[4]:


def autoregr(X, param):
    d = param[:,:3]
    X = np.matmul(d, X) + param[:,-1] + np.random.normal(size=[3], scale=0.2)
    return X

def autoregr_tf(X, param):
    d = param[:,:3]
    X = tf.matmul(d, X) + param[:,-1]
    return X


# In[5]:


xs = [np.random.normal(size=dim[0])]
for i in range(n-1):
    xs.append(autoregr(xs[i], params[i]))
xs = np.array(xs)[np.newaxis,:].astype('float32')


# In[6]:


def create_step_flow(name=None, prev_flow=None):
    with tf.variable_scope(name, reuse=None):
        if prev_flow is not None:
            aux_vars = prev_flow.output
        else:
            aux_vars = None
            
        step_flow = DFlow([NVPFlow(dim=dim[0]*dim[1], name='nvp{}'.format(i), aux_vars=aux_vars)                           for i in range(4)])
    return step_flow


# In[7]:


flows = [create_step_flow('step_flow0')]
for i in range(n-1):
    new = create_step_flow('step_flow' + str(i+1), flows[-1])
    flows.append(new)


# In[8]:


outputs = tf.concat([x.output for x in flows], axis=0)[tf.newaxis]


# In[9]:


prior = PWalk.logdens(outputs)


# In[10]:


#outputs = tf.cast(outputs, tf.float64)


# In[11]:


prior


# In[12]:


def create_loglik():
    with tf.name_scope('loglik'):
        obs_d = Normal(dim=None, sigma=0.2, mu=0)
        out = tf.reshape(outputs, [n, 3, 4])
        
        ll = 0
        for i in range(n-1):
            pred = xs[0,i+1] - autoregr_tf(xs[0,i][:,np.newaxis], out[i])
            ll += obs_d.logdens(pred)
    return ll


# In[13]:


logl = create_loglik()


# In[14]:


xs.shape


# In[15]:


logl


# In[16]:


with tf.name_scope('ent'):
    ent = sum([flow.logdens for flow in flows])


# In[17]:


ent


# In[18]:


loss = tf.identity(-logl - prior + ent, name='loss')


# In[19]:


opt = tf.train.AdamOptimizer().minimize(loss)


# In[20]:


sess = tf.InteractiveSession()
sess = swrap(sess)
tf.global_variables_initializer().run()

writer = tf.summary.FileWriter('/tmp/tfdbg')
writer.add_graph(tf.get_default_graph())

# In[21]:


for _ in range(10):
    for _ in range(1000):
        l, _ = sess.run([loss, opt], {phase:True})
    print(l)

