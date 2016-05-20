#!/bin/sh
#$ -S /bin/sh
#$ -cwd
#$ -V
#$ -m e
#$ -M ericmjl@mit.edu
#$ -pe whole_nodes 1
#############################################

qsub nnet_synth0.sh
qsub nnet_synth1.sh
qsub nnet_synth2.sh
qsub nnet_synth3.sh
qsub nnet_synth4.sh
qsub nnet_synth5.sh
qsub nnet_synth6.sh
qsub nnet_synth7.sh
qsub nnet_synth8.sh
qsub nnet_synth9.sh
qsub nnet_synth10.sh
qsub nnet_synth11.sh
qsub nnet_synth12.sh
qsub nnet_synth13.sh
qsub nnet_synth14.sh
qsub nnet_synth15.sh
qsub nnet_synth16.sh
qsub nnet_synth17.sh
