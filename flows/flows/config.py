import tensorflow as tf

floatX = 'float64'
phase = tf.placeholder_with_default(True, shape=(), name='learning_phase')
