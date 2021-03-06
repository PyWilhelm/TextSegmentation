#!/bin/bash

#BSUB -J "TextSegmentation experience"
#BSUB -u ziyang.li.nk@gmail.com
#BSUB -N
#BSUB -eo /home/kurse/oe14ireg/TextSegmentation/experience/stderr.txt
#BSUB -oo /home/kurse/oe14ireg/TextSegmentation/experience/stdout.txt
#BSUB -n 1
#BSUB -M 10240
#BSUB -W 10:00
#BSUB -q kurs3
#BSUB -x

cd /home/kurse/oe14ireg/TextSegmentation/experience/
THEANO_FLAGS='mode=FAST_RUN,device=gpu,force_device=True,floatX=float32' python3 /home/kurse/oe14ireg/TextSegmentation/experience/LSTM_bitmap.py training_x training_y LSTM_bitmap5.model
THEANO_FLAGS='mode=FAST_RUN,device=gpu,force_device=True,floatX=float32' python3 /home/kurse/oe14ireg/TextSegmentation/experience/LSTM_multilayers.py training_x training_y LSTM_multi5.model
THEANO_FLAGS='mode=FAST_RUN,device=gpu,force_device=True,floatX=float32' python3 /home/kurse/oe14ireg/TextSegmentation/experience/GRN_bitmap.py training_x training_y GRN_bitmap5.model
THEANO_FLAGS='mode=FAST_RUN,device=gpu,force_device=True,floatX=float32' python3 /home/kurse/oe14ireg/TextSegmentation/experience/GRN_multilayer.py training_x training_y GRN_multi5.model
THEANO_FLAGS='mode=FAST_RUN,device=gpu,force_device=True,floatX=float32' python3 /home/kurse/oe14ireg/TextSegmentation/experience/BiLSTM_bitmap.py training_x training_y BiLSTM_bitmap5.model

echo lstmb1
THEANO_FLAGS='mode=FAST_RUN,device=gpu,force_device=True,floatX=float32' python3 /home/kurse/oe14ireg/TextSegmentation/experience/evaluate.py small_x small_y lstmb1

echo lstmbm
THEANO_FLAGS='mode=FAST_RUN,device=gpu,force_device=True,floatX=float32' python3 /home/kurse/oe14ireg/TextSegmentation/experience/evaluate.py small_x small_y lstmbm

echo grnb1
THEANO_FLAGS='mode=FAST_RUN,device=gpu,force_device=True,floatX=float32' python3 /home/kurse/oe14ireg/TextSegmentation/experience/evaluate.py small_x small_y grnb1

echo grnbm
THEANO_FLAGS='mode=FAST_RUN,device=gpu,force_device=True,floatX=float32' python3 /home/kurse/oe14ireg/TextSegmentation/experience/evaluate.py small_x small_y grnbm

echo bilstm
THEANO_FLAGS='mode=FAST_RUN,device=gpu,force_device=True,floatX=float32' python3 /home/kurse/oe14ireg/TextSegmentation/experience/evaluate.py small_x small_y bilstmb
