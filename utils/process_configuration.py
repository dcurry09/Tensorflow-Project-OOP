"""
Parse the JSON configuration file of the experiment.

Configuration file holds the parameters to intialize the NN model.
These files are located in configaration_files folder.

@version: 1.0
"""

import json
import os
#from bunch import Bunch

class ConfigurationParameters:

        def __init__(self, args):
                """
	        Intialize the data members.
                
	        :param json_file: Path to the JSON configuration file.
	        :return none
	        :raises none
	        """

                
                self.args = args

                json_file = self.args['config']

	        # Parse the configurations from the config json file provided.
                with open(json_file, 'r') as config_file:
                        self.config_dictionary = json.load(config_file)
                
	        # Convert the dictionary to a namespace using bunch library.
                #self.config_namespace = Bunch( self.config_dictionary )
                self.config_namespace = self.config_dictionary

	        # Process the configuration parameters.
                self.process_config()
                
                return


        def process_config(self):
                """
	        Processes the configarion parameters of the NN experiment.
                
	        :param none
	        :return none
	        :raises none
                """

                # Saved-Model directory
                self.config_namespace['saved_model_dir'] = os.path.join("./experiments/", self.config_namespace['exp_name'], "saved_models/")

                # Graph directory.
                self.config_namespace['graph_dir'] = os.path.join("./experiments/", self.config_namespace['exp_name'], "graphs/")

                # Create the above directories.
                self.create_dirs( [self.config_namespace['graph_dir'], self.config_namespace['saved_model_dir']] )
                
                return

        def create_dirs(self, dirs):
                """
	        Creates a directory structure for Graphs and Images generated during the run of the experiment.
                
	        :param dirs: a list of directories to create if these directories are not found
	        :return exit_code: 0:success -1:failed
                :raises none
                """

                try:
                        for d in dirs:
                                if not os.path.exists(d):
                                        os.makedirs(d)
                                        return 0
                
                except Exception as err:
                        print( "Creating directories error: {0}".format(err) )
                        exit(-1)
