cd code
source setenv.sh
cd -

#rm checks/* jobs/* npys/* slurm_out/*
mkdir checks jobs npys slurm_out

$MYPYTHON ${CODEDIR}/submit_monitor.py
$MYPYTHON ${CODEDIR}/eval_models.py
