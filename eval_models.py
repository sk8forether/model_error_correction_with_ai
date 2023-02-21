from check_model import sub_collect_models, sub_eval_model, sub_saliency
import numpy as np

import time
import logging
logging.basicConfig(level=logging.INFO)

f = open('evaluate_model.test','w')

def execute(if_renew):
  for var in ['t','q','u','v']: # loop through variables
    time.sleep(5)

    filename = f"checks/conv2d_tpsuvq_online_{var}_3_1_4096_3_0.25_8_mse_0.0001_0.05_sub"
#    filename = df[(df.vars_out==var) & (df.trunc=='sub') & 
#         ars_sfc=='online')].sort_values('valid_min').iloc[0].filename # find model that has minimal validation loss
    logging.info(var)
    logging.info(filename)
    if if_renew:
        sub_eval_model(filename,if_renew=True,if_wait=False) # submit evaluation
        #print(filename)
    else:
        y_pred, y=sub_eval_model(filename,if_renew=False,if_wait=True) # read from previous evaluation (if exists) and output
        f.write("Variable {var} MSE: {mse:.5f}\n".format(var=var, mse=np.mean((y-y_pred)**2)))

execute(True)
execute(False)


