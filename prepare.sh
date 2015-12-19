#!/bin/bash

#BSUB -J "deepnlp sample program"
#BSUB -eo /home/kurse/oe14ireg/deepnlp/sample_program/stderr.txt
#BSUB -oo /home/kurse/oe14ireg/deepnlp/sample_program/stdout.txt
#BSUB -n 1
#BSUB -M 2048
#BSUB -W 60
#BSUB -x
#BSUB -q kurs3

cd /home/kurse/oe14ireg/TextSegmentation
THEANO_FLAGS='mode=FAST_RUN,device=gpu,force_device=True,floatX=float32' python3 /home/kurse/oe14ireg/TextSegmentation/prepare_data.py 2.bz2 training
THEANO_FLAGS='mode=FAST_RUN,device=gpu,force_device=True,floatX=float32' python3 /home/kurse/oe14ireg/TextSegmentation/prepare_data.py 3.bz2 test