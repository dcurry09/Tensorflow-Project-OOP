#!/usr/bin/env python3
"""
Implements a TF trainer class inherited from base_train super class.

@author: David Curry
@version: 1.0
"""

from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np


class MnistTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        """
        Constructor to initialize the trainer object
        
        :param config: the json configuration namespace.
        :return none
        :raises none
        """
        super(MnistTrainer, self).__init__(sess, model, data, config, logger)

        # finally setup the initialisation operator
        #sess.run(self.model.init_op)
        
    def train_epoch(self):
        """
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary        
        :param none
        :return none
        :raises none
        """

        # create a loop object with # of total batches
        total_batch = int(len(self.data.mnist.train.labels)/self.config['batch_size'])
        loop = tqdm(range(total_batch))
        
        # store current loss and acc
        losses = []
        #accs = []
        
        # loop over # of batches in an epoch
        for _ in loop:
            cost = self.train_step()
            losses.append(cost)
            #accs.append(acc)

        # save aggregate metrics over all batches in the current epoch
        loss = np.mean(losses)
        #acc = np.mean(accs)

        # helps with full training summary
        cur_it = self.model.global_step_tensor.eval(self.sess)
        
        summaries_dict = {'loss': loss}
                                                    
        print("Cost =", "{:.3f}".format(loss))
            
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)
        
    def train_step(self):
        """
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        
        # create batches from generator
        batch_x, batch_y = self.data.next_batch( batch_size = self.config['batch_size'] )
        
        # Use TFs feed_dict object
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y}
        
        # train a batch of data
        _, cost = self.sess.run([self.model.optimiser, self.model.loss], feed_dict=feed_dict)
        #print(cost)
        return cost
    
