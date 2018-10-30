#!/usr/bin/env python3
"""
Implements a TF Model class by inheriting the Model base class.

@author: David Curry
@version: 1.0
"""

from base.base_model import BaseModel
import tensorflow as tf


class ImdbModel(BaseModel):

    def __init__(self, config):
        """
        Constructor to initialize the TF model class by inheritance from super.
        
        :param config
        :return none
        :raises none
        """
        super(ImdbModel, self).__init__(config)
        self.build_model()
        self.init_saver()


        
    def build_model(self):
        """
        Build the Tensorflow model
        
        :param self
        :return none
        :raises none
        """
        
        self.is_training = tf.placeholder(tf.bool)
        
        self.x = tf.placeholder(tf.float32, shape=[None] + self.config['state_size'])
        self.y = tf.placeholder(tf.float32, shape=[None, 10])
        
        # network architecture
        d1 = tf.layers.dense(self.x, 512, activation=tf.nn.relu, name="dense1")
        d2 = tf.layers.dense(d1, 10, name="dense2")
        
        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=d2))
            self.train_step = tf.train.AdamOptimizer(self.config['learning_rate']).minimize(self.cross_entropy,
                                                                                            global_step=self.global_step_tensor)
            correct_prediction = tf.equal(tf.argmax(d2, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def init_saver(self):
        """
        Initialize the tensorflow saver that will be used in saving the checkpoints.

        :param self
        :return none
        :raises none
        """
        self.saver = tf.train.Saver(max_to_keep=self.config['max_to_keep'])

