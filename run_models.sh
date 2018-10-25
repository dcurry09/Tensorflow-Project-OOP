#!/bin/sh

# Setup the execution path.
export PYTHONPATH=`pwd`:$PYTHONPATH

# Function to display the help message.
usage() {
 
	echo "Usage: $0 -x, --model <string> [-c, --config <path>] [-e, --epoch <number>] [-h,--help]"
	echo	
	echo "Runs the Keras NN Model"
	echo	
	echo "Mandatory or optional arguments to long options are also mandatory or optional for any corresponding short options."
	echo
	echo "Model options:"
	echo "-x, --model name of the model to be run."
	echo "				As of now, acceptable values are:"
	echo "				bankH_classifier for ank H sequence loan classification"
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
    echo "-x, --model is compulsory."
    echo
    usage
fi

# Argument variables.
EXP=
CONFIG=
EPOCH=

# Parse the command line arguments.
ARGS=`getopt -o hx:c:e: --long help,model:,config:,epoch: -n 'run_models.sh' -- "$@"`
eval set -- "$ARGS"

while true; do
  case "$1" in
    -x | --model ) EXP=$2; shift 2 ;;
    -c | --config) CONFIG=$2; shift 2;;
    -e | --epoch ) EPOCH=$2; shift 2;;
    -h | --help ) usage; exit 0 ;;
    -- ) shift; break ;;
    * ) usage; exit 1 ;;
  esac
done
 
# # Check for -model argument.
if [ -z $EXP ]
then
    echo "-x, --model is compulsory."
    echo 
    usage
fi

# Run the model with required arguments.
if [ "$EXP" = "bankH_classifier" ] && [ ! -z $CONFIG ] && [ ! -z $EPOCH ]
then
	echo "Executing bankH_seq_classifier model with config file and epoch arguments."
	python ./mains/bankH_main.py -c $CONFIG -e $EPOCH
elif [ "$EXP" = "bankH_classifier" ] && [ ! -z $EPOCH ]
then
	echo "Executing bankH_seq_classifier model with epoch argument."
	python ./mains/bankH_main.py -e $EPOCH
elif [ "$EXP" = "bankH_classifier" ] && [ ! -z $CONFIG ]
then
	echo "Executing bankH_seq_classifier model with config argument."
	python ./mains/bankH_main.py -c $CONFIG
elif [ "$EXP" = "bankH_classifier" ] 
then
	echo "Executing bankH_seq_classifier model."
	python ./mains/example.py 
fi
