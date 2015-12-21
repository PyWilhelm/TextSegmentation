#!/bin/bash

#BSUB -J "TextSegmentation experience"
#BSUB -u ziyang.li.nk@gmail.com
#BSUB -N
#BSUB -eo /home/kurse/oe14ireg/TextSegmentation/experience/stderr.txt
#BSUB -oo /home/kurse/oe14ireg/TextSegmentation/experience/stdout.txt
#BSUB -n 1
#BSUB -M 30480
#BSUB -W 24:00
#BSUB -x
#BSUB -q kurs3

cd /home/kurse/oe14ireg/TextSegmentation/experience/
THEANO_FLAGS='mode=FAST_RUN,device=gpu,force_device=True,floatX=float32' python3 /home/kurse/oe14ireg/TextSegmentation/experience/LSTM_bitmap.py test1_x test1_y LSTM_bitmap.model