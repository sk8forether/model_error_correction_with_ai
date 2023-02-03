########## remove hardcoding of paths etc ##########
training.py --- ddd='/scratch2/NCEPDEV/stmp1/Tse-chun.Chen/anal_inc/npys/ifs' # dataset location
submit_monitor.py --- userid, main_dir, python_exe, sbatch

########## add new training/test options ##########

check_model.py
  get_test_dataset and get_train_dataset add new hyperparameter test_slice and train_valid_slice or idx_include

preprocess.py -- add subsampling later

training.py -- add the same logic as check_model.py get_test_dataset

submit_monitor.py 
  params_space -- specify new slices

