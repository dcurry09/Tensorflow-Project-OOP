"""
Execution Flow for the IMDB Sequence Classifier

@author: David Curry
@version: 1.0
"""

import tensorflow as tf

from data_loader.imdb_data_loader import ImdbDataLoader
from models.imdb_model import ImdbModel
from trainers.imdb_trainer import ImdbTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args['config'])
        
    except:
        print("missing or invalid arguments")
        exit(0)
        
    # create the experiments dirs
    create_dirs([config['summary_dir'], config['checkpoint_dir']])
    
    # create tensorflow session
    sess = tf.Session()
    
    # create your data generator
    data = ImdbDataLoader(config)
    
    # create an instance of the model you want
    model = ImdbModel(config)
    
    # create tensorboard logger
    logger = Logger(sess, config)
    
    # create trainer and pass all the previous components to it
    trainer = ImdbTrainer(sess, model, data, config, logger)

    #load model if exists
    model.load(sess)

    # here you train your model
    trainer.train()


if __name__ == '__main__':
    main()
