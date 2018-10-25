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
    
	parser = argparse.ArgumentParser( description = __doc__ )

	# Configuration file path argument.
	parser.add_argument(
		'-c', '--config',
		metavar = 'C',
		help = 'The Configuration file',
		default = None,
		required = False
	)

	# Convert to dictonary.
	args = vars(parser.parse_args())

	print('Using configurations from file:', args['config'])

	return args
