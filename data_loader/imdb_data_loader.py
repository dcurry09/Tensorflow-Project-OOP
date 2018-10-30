#!/usr/bin/env python3
"""
Implements a data loader class by inheriting the DataLoader base class.

IMDB dataset is accessed through the Keras API

@author: David Curry
@version: 1.0
"""

from base.data_loader_base import DataLoader
from keras.datasets import imdb
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import keras as keras
import os
import pandas as pd
import numpy as np

class ImdbDataLoader(DataLoader):
        
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
                
                print('Loading the IMDB dataset from Keras API...')
                
                # num of words in corpus to use
                self.top_words = 5000
                (self.train_data, self.train_labels), (self.test_data, self.test_labels) = imdb.load_data(num_words = self.top_words)
                
        
        def next_batch(self, batch_size):
                """
                Create generator with yield function.  
                
                :param batch size
                :returns data from generator of size batch 
                :raises none
                """
                idx = np.random.choice(len(self.train_data), batch_size)
                yield self.train_data[idx], self.train_labels[idx]
                
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

                # truncate and pad input sequences
                self.max_review_length = 500
                self.train_data = keras.preprocessing.sequence.pad_sequences(self.train_data, maxlen = self.max_review_length)
                self.test_data  = keras.preprocessing.sequence.pad_sequences(self.test_data, maxlen = self.max_review_length)
                #print('\nself.test_data:', self.test_data[:5])

                # normalize data range to 0-1
                # scaler = MinMaxScaler(feature_range=(0, 1))
                # scaler = scaler.fit(self.train_data)
                # self.train_data = scaler.transform(self.train_data)
                # scaler = scaler.fit(self.test_data)
                # self.test_data = scaler.transform(self.test_data)
                #print('normed self.test_data:', self.test_data[:5])
                
                
                # LSTM requires data to have 3 dimensions
                # https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
                # 1) Samples. One sequence is one sample. A batch is comprised of one or more samples.
                # 2) Time Steps. One time step is one point of observation in the sample.
                # 3) Features. One feature is one observation at a time step.
                #self.train_data = np.reshape(self.train_data, (self.train_data.shape[0], self.train_data.shape[1], 1))
                #self.test_data  = np.reshape(self.test_data, (self.test_data.shape[0], self.test_data.shape[1], 1))
                
                # Convert the class labels from categorical to boolean one hot encoded vector.
                #self.train_label_one_hot = keras.utils.to_categorical( self.train_labels )
                #self.test_label_one_hot  = keras.utils.to_categorical( self.test_labels )
                #self.train_label_one_hot = self.train_labels.values
                #self.test_label_one_hot  = self.test_labels.values
                
                print("\nTraining and testing datasets are loaded and ready for training...\n")
                return
