#!/usr/bin/env python3
"""
Process the json configuration file.

Configuration file holds the parameters to intialize the NN model.
These files are located in configaration_files folder.
"""

import argparse


def get_args():
    """
    Get arguments from the command line.
    
    :param none
    :return none
    :raises none
    """

    argparser = argparse.ArgumentParser(description=__doc__)

    # Configuration file path argument.
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')

    # Convert to dictonary.
    args = vars(argparser.parse_args())
    #args = argparser.parse_args()
    print('get_args:', args)

    return args





















