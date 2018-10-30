#!/usr/bin/env python3
"""
Creates a base class for training of TensorFlow models.

@author: David Curry
@version: 1.0
"""

import tensorflow as tf


class BaseTrain:

    def __init__(self, sess, model, data, config, logger):
        """
        Constructor to initialize the TF trainer
        :param config: the json configuration namespace.
        :return none
        :raises none
        """

        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        self.data = data
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

    def train(self):
        """
        train TF model over all epochs (and batches).
        :param self
        :return none
        :raises none
        """
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config['num_epochs'] + 1, 1):
            self.train_epoch()
            self.sess.run(self.model.increment_cur_epoch_tensor)
        
        print(self.sess.run(self.model.accuracy, feed_dict={self.model.x: self.data.mnist.test.images, self.model.y: self.data.mnist.test.labels}))
    
    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_step(self):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError
