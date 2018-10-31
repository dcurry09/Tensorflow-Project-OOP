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
        
        NOTICE:
        We have to pad each sequence to reach 'max_seq_len' for TensorFlow
        consistency (we cannot feed a numpy array with inconsistent
        dimensions). The dynamic calculation will then be perform thanks to
        'seqlen' attribute that records every actual sequence length.

        :param self
        :return none
        :raises none
        """
        
        self.is_training = tf.placeholder(tf.bool)

        batch_size  = self.config['batch_size']
        max_seq_len = self.config['max_seq_len']
        lstmUnits   = self.config['n_hidden']
        
        # Graph Input
        self.x = tf.placeholder(tf.int32, shape=[None, max_seq_len])
        self.y = tf.placeholder(tf.int32, shape=[None, self.config['num_classes']])
        #self.x = tf.placeholder(tf.int32, shape=[batch_size, max_seq_len])
        #self.y = tf.placeholder(tf.int32, shape=[batch_size, self.config['num_classes']])
        
        # embedding layer
        word_embeddings = tf.get_variable("word_embeddings", [self.config['vocabulary_size'], self.config['embedding_size']])
        embedded_data = tf.nn.embedding_lookup(word_embeddings, self.x)

        # build the LSTM cell
        lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
        #lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
        value, _ = tf.nn.dynamic_rnn(lstmCell, embedded_data, dtype=tf.float32)

        weight = tf.Variable(tf.truncated_normal([lstmUnits, self.config['num_classes']]))
        bias = tf.Variable(tf.constant(0.1, shape=[self.config['num_classes']]))
        value = tf.transpose(value, [1, 0, 2])
        last = tf.gather(value, int(value.get_shape()[0]) - 1)
        self.y_ = (tf.matmul(last, weight) + bias)

        # define loss and gradient descent technique
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
        
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

