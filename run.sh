cd code
source setenv.sh
cd -

mkdir checks/ npys

$MYPYTHON code/submit_monitor.py
$MYPYTHON code/eval_models.py
