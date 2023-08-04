#!/bin/sh
#SBATCH --nodes 1
#SBATCH -A gsienkf
#SBATCH -p fgewf
#SBATCH --qos windfall
#SBATCH -t 8:0:0

#salloc -t 8:0:0 -A gsienkf -p fgewf --qos=windfall -N 1


cd /home/Sergey.Frolov/work/model_error/work/stefan_replay
echo $PWD


cd code
echo $PWD
source ./setenv.sh
cd -

#$GPUPYTHON code/sequential_training.py
$GPUPYTHON batch_training.py
 

