#!/usr/bin/env python3
"""
Implements a TF Model class by inheriting the Model base class.

@author: David Curry
@version: 1.0
"""

from base.base_model import BaseModel
import tensorflow as tf

class MnistModel(BaseModel):

    def __init__(self, config):
        """
        Constructor to initialize the TF model class by inheritance from super.
        
        :param config
        :return none
        :raises none
        """
        super(MnistModel, self).__init__(config)
        self.build_model()
        self.init_saver()


        
    def build_model(self):
        """
        Build the Tensorflow model
        
        :param self
        :return none
        :raises none
        """

        batch_size = self.config['batch_size']
        
        self.is_training = tf.placeholder(tf.bool)
        
        # declare the training data placeholders
        # input x - for 28 x 28 pixels = 784
        self.x = tf.placeholder(tf.float32, [None, 784])

        # now declare the output data placeholder - 10 digits
        self.y = tf.placeholder(tf.float32, [None, 10])

        # now declare the weights connecting the input to the hidden layer
        self.W1 = tf.Variable(tf.random_normal([784, 300], stddev=0.03), name='W1')
        self.b1 = tf.Variable(tf.random_normal([300]), name='b1')

        # and the weights connecting the hidden layer to the output layer
        self.W2 = tf.Variable(tf.random_normal([300, 10], stddev=0.03), name='W2')
        self.b2 = tf.Variable(tf.random_normal([10]), name='b2')

        # calculate the output of the hidden layer
        self.hidden_out = tf.add(tf.matmul(self.x, self.W1), self.b1)
        self.hidden_out = tf.nn.relu(self.hidden_out)
        
        # now calculate the hidden layer output - in this case, let's use a softmax activated output layer
        self.y_ = tf.nn.softmax(tf.add(tf.matmul(self.hidden_out, self.W2), self.b2))
        
        # define the loss function
        self.y_clipped = tf.clip_by_value(self.y_, 1e-10, 0.9999999)
        self.cross_entropy = -tf.reduce_mean(tf.reduce_sum(self.y * tf.log(self.y_clipped) + (1 - self.y) * tf.log(1 - self.y_clipped), axis=1))
        
        # add an optimiser
        self.optimiser = tf.train.GradientDescentOptimizer(learning_rate=self.config['learning_rate']).minimize(self.cross_entropy)

        # define an accuracy assessment operation
        self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        
        

    def init_saver(self):
        """
        Initialize the tensorflow saver that will be used in saving the checkpoints.

        :param self
        :return none
        :raises none
        """
        self.saver = tf.train.Saver(max_to_keep=self.config['max_to_keep'])

