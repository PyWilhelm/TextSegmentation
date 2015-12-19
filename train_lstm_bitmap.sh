#!/bin/bash

#BSUB -J "deepnlp sample program"
#BSUB -eo /home/kurse/oe14ireg/deepnlp/sample_program/stderr.txt
#BSUB -oo /home/kurse/oe14ireg/deepnlp/sample_program/stdout.txt
#BSUB -n 1
#BSUB -M 2048
#BSUB -W 60
#BSUB -x
#BSUB -q kurs3

cd /home/kurse/oe14ireg/TextSegmentation/experience/
THEANO_FLAGS='mode=FAST_RUN,device=gpu,force_device=True,floatX=float32' python3 /home/kurse/oe14ireg/TextSegmentation/experience/LSTM_bitmap.py training_x training_y LSTM_bitmap.model