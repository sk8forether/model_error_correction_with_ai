from check_model import sub_collect_models, sub_eval_model, sub_saliency
import numpy as np

import time
import logging
logging.basicConfig(level=logging.INFO)

def make_filename(var):
  import json
  with open('params_space.dict') as json_file:
    ps=json.loads(json_file.read())
  vars_f06 = ps["vars_f06"][0]
  vars_sfc = ps["vars_sfc"][0]
  testset = ps["testset"][0]
  kernel_sizes = ps["kernel_sizes"][0]
  channels = ps["channels"][0]
  n_conv = ps["n_conv"][0]
  p = ps["p"][0]
  bs = ps["bs"][0]
  loss = ps["loss"][0]
  lr = ps["lr"][0]
  wd = ps["wd"][0]
  etd = ps["end_of_training_day"][0]
  tvld = ps["training_validation_length_days"][0]
  tvr = ps["training_to_validation_ratio"][0]
  filename = f"conv2d_{vars_f06}_{vars_sfc}_{var}_{testset}_{kernel_sizes}_{channels}_{n_conv}_{p}_{bs}_{loss}_{lr}_{wd}_sub_{etd}_{tvld}_{tvr}"
  return filename


f = open('evaluate_model.test','w')

def execute(if_renew):
  for var in ['t','q','u','v']: # loop through variables
#  for var in ['q']: # loop through variables

    filename = "checks/"+make_filename(var)
#    filename = df[(df.vars_out==var) & (df.trunc=='sub') & 
#         ars_sfc=='online')].sort_values('valid_min').iloc[0].filename # find model that has minimal validation loss
    logging.info(var)
    logging.info(filename)
    if if_renew:
        sub_eval_model(filename,if_renew=True,if_wait=False) # submit evaluation
        #print(filename)
    else:
        y_pred, y=sub_eval_model(filename,if_renew=False,if_wait=True) # read from previous evaluation (if exists) and output
        f.write("Variable {var} MSE: {mse:.2f}\n".format(var=var, mse=np.mean((y-y_pred)**2)))
    time.sleep(10)

#execute(True)
execute(False)


