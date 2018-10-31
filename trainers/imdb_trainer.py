#!/usr/bin/env python3
"""
Implements a TF trainer class inherited from base_train super class.

@author: David Curry
@version: 1.0
"""

from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np


class ImdbTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        """
        Constructor to initialize the training and testing datasets for Bank H Sequences
        
        :param config: the json configuration namespace.
        :return none
        :raises none
        """
        super(ImdbTrainer, self).__init__(sess, model, data, config, logger)
    
    def train_epoch(self):
        """
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary        
        :param none
        :return none
        :raises none
        """

        # create a loop object with size of batch
        total_batch = int(len(self.data.train_labels)/self.config['batch_size'])
        loop = tqdm(range(total_batch))
        
        # store current loss and acc
        losses = []
        accs = []

        # loop over # of batches in an epoch
        for _ in loop:

            # train and get the loss for each batch
            acc, loss = self.train_step()
            losses.append(loss)
            accs.append(acc)
            
        # save aggregate metrics over all batches in the current epoch
        loss = np.mean(losses)
        acc = np.mean(accs)
        print('Cost:', loss, ' -- Acc:', acc)
                
        # summary epoch checkpoint
        epoch_count = self.model.cur_epoch_tensor.eval(self.sess)

        # for logger class
        summaries_dict = {'loss': loss, 'acc': acc}

        self.logger.summarize(epoch_count, summaries_dict=summaries_dict)

        self.model.save(self.sess)
        
    def train_step(self):
        """
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        
        # create batches from generator
        batch_x, batch_y = next(self.data.next_batch(self.config['batch_size']))

        # Use TFs feed_dict object
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y}
        
        # train a batch of data
        acc, _, cost = self.sess.run([self.model.accuracy, self.model.optimizer, self.model.loss], feed_dict=feed_dict)

        return acc, cost
