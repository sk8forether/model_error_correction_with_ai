#!/bin/sh
#SBATCH -N 1
#SBATCH -A gsienkf
#SBATCH -p bigmem
#SBATCH --time 03:59:00

cd /home/Sergey.Frolov/work/model_error/work/sliding_window

cd code
source ./setenv.sh
cd -

#$GPUPYTHON code/sequential_training.py
$MYPYTHON code/sequential_evaluation.py
 

