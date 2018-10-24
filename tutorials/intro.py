###############################
# intro.py
# A simple script to outline the
# basic Tensorflow operations.
#
# @source:  http://adventuresinmachinelearning.com/stochastic-gradient-descent/
# @author: David Curry
# @created: 10/01/2018
#
##############################

import tensorflow as tf
import numpy as np


####### Create a simple equation: a = (b+c) * (c+2) 

# first, create a TensorFlow constant
const = tf.constant(2.0, name="const")

# create TensorFlow variables
b = tf.placeholder(tf.float32, [None, 1], name='b')
c = tf.Variable(1.0, name='c')

# now create some operations: d=b+c, e=c+2, a=d*e
d = tf.add(b, c, name='d')
e = tf.add(c, const, name='e')
a = tf.multiply(d, e, name='a')

# setup the variable initialisation
init_op = tf.global_variables_initializer()

# If b was a tensor to be filled (like hidden layer weights) we need to create a placeholder.

# create TensorFlow variables


# start the session
with tf.Session() as sess:

    # initialise the variables
    sess.run(init_op)

    # compute the output of the graph
    a_out = sess.run(a, feed_dict={b: np.arange(0, 10)[:, np.newaxis]})
    print("Variable a is {}".format(a_out))


