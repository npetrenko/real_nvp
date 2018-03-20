from __future__ import print_function
import tensorflow as tf
from tensorflow.python import debug as tf_dbg
import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

wrapper = tf_dbg.LocalCLIDebugWrapperSession

def write_graph():
    writer = tf.summary.FileWriter('/tmp/tfdbg')
    writer.add_graph(tf.get_default_graph())
