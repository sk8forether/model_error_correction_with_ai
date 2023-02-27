cd code
source setenv.sh
cd -

#rm checks/* jobs/* npys/* slurm_out/*
mkdir checks jobs npys slurm_out

$GPUPYTHON ${CODEDIR}/submit_monitor.py

# run this after training is complete
#$MYPYTHON ${CODEDIR}/eval_models.py
