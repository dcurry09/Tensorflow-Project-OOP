#!/usr/bin/env python3
"""
Implements a data loader class by inheriting the DataLoader base class.

MNIST dataset is accessed through the Keras API

@author: David Curry
@version: 1.0
"""

from base.data_loader_base import DataLoader
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import keras as keras
import os
import pandas as pd
import numpy as np

class MnistDataLoader(DataLoader):
        
        def __init__(self, config):
                """
		Constructor to initialize the training and testing datasets for Bank H Sequences
                
		:param config: the json configuration namespace.
		:return none
		:raises none
		"""
                
                super().__init__(config)
                return
		
        def load_dataset(self):
                """
		Loads the Bank H sequence dataset

		:param none
		:return none
		:raises none
		"""
                
                print('Loading the MNIST dataset from Keras API...')
                self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
                
        
        def next_batch(self, batch_size):
                """
                Create generator with yield function.  
                
                :param batch size
                :returns data from generator of size batch 
                :raises none
                """
                # not needed for mnist
                return
                
        def preprocess_dataset(self):
                """
                Preprocess the sequence dataset.
                
                Pad if necessary.

                Create integer sequences and an associated vocabulary.

		Converts the categorical class labels to boolean one-hot encoded vector for training and testing datasets.
                
		:param none
		:returns none
		:raises none
		"""
                # not needed for mnist
                return
