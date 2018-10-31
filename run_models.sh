#!/bin/sh

# Setup the execution path.
export PYTHONPATH=`pwd`:$PYTHONPATH

# Function to display the help message.
usage() {
 
	echo "Usage: $0 -m, --model <string> [-c, --config <path>] [-e, --epoch <number>] [-h,--help]"
	echo	
	echo "Runs the Tensorflow NN Model"
	echo	
	echo "Mandatory or optional arguments to long options are also mandatory or optional for any corresponding short options."
	echo
	echo "Model options:"
	echo "-m, --model name of the model to be run."
	echo "				As of now, acceptable values are:"
	echo "                          * imdb_classifier for IMDB sequence classification"
	echo "                          * mnist for image classification"
	echo "-c, --config		use this configuration file." 
	echo "-e, --epoch		number of training epoches."
	echo
	echo "Other options:"
	echo "-h, --help		display this help and exit."
}

# Check for mandatory arguments.
if [ $# -eq 0 ]
then
    echo "No arguments supplied."
    echo "-m, --model is compulsory."
    echo "-c, --config is compulsory."
    echo
    usage
fi

# Argument variables.
EXP=
CONFIG=
EPOCH=

# Parse the command line arguments.
ARGS=`getopt -o hm:c:e: --long help,model:,config:,epoch: -n 'run_models.sh' -- "$@"`
eval set -- "$ARGS"

while true; do
  case "$1" in
    -m | --model ) EXP=$2; shift 2 ;;
    -c | --config) CONFIG=$2; shift 2;;
    -e | --epoch ) EPOCH=$2; shift 2;;
    -h | --help ) usage; exit 0 ;;
    -- ) shift; break ;;
    * ) usage; exit 1 ;;
  esac
done
 
# # Check for -model argument.
if [ -z $EXP ] && [ ! -z $CONFIG ]
then
    echo "-m, --model is compulsory."
    echo "-c, --config is compulsory."
    echo 
    #usage
fi

# Run the model with required arguments.
if [ "$EXP" = "imdb_classifier" ] && [ ! -z $CONFIG ]
then
	echo "Executing imdb_classifier model with config argument."
	python ./mains/imdb_main.py -c $CONFIG

elif [ "$EXP" = "mnist" ] && [ ! -z $CONFIG ]
then
    echo "Executing MNIST classifier model with config argument."
    python ./mains/mnist_main.py -c $CONFIG
fi

