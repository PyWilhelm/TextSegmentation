#!/bin/bash

#BSUB -J "test cpu"
#BSUB -eo /home/kurse/oe14ireg/TextSegmentation/stderr_cpu
#BSUB -oo /home/kurse/oe14ireg/TextSegmentation/stdout_cpu
#BSUB -n 1
#BSUB -M 2048
#BSUB -W 60
#BSUB -x
#BSUB -q kurs3

cd /home/kurse/oe14ireg/TextSegmentation
python3 /home/kurse/oe14ireg/TextSegmentation/test_cpu.py