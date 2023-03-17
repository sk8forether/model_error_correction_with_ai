#!/scratch1/NCEPDEV/global/Tse-chun.Chen/anaconda3/envs/ltn/bin/python
# salloc --account=gsienkf --partition=bigmem --time=00:59:00 --nodes=1 --ntasks=40
import torch
from training import create_checkpoint_filename
from check_model import read_model, get_test_dataset
import numpy as np
import os

torch.set_num_threads(40)

#updates parameter tuple and resets the network
def increment_parameter_tuple(p_before):
  fn_before = create_checkpoint_filename(p_before)
  p_new=list(p_before)
  p_new[start_day_postion_in_the_list]=p_new[start_day_postion_in_the_list]+training_step_length
  p_new=tuple(p_new)
  fn_new = create_checkpoint_filename(p_new)
  return p_new

# evaluate model
def my_eval_model(model, test_Loader):
  with torch.set_grad_enabled(False):
    model.eval()
    for X, y in test_Loader:
      y_pred = model(X)
  return y_pred, y

def compute_skill(y, y_pred):
   y_pred_ts=y_pred.cpu().detach().numpy().view().reshape((y.shape[0], np.prod(y.shape[1:])))
   y_ts=y.detach().numpy().view().reshape((y.shape[0], np.prod(y.shape[1:])))
   skill = 1-np.mean((y_pred_ts-y_ts)**2,axis=1)/np.mean((y_ts)**2,axis=1)
   return skill


#define sequential training configuration
training_step_length=7
start_day_postion_in_the_list=-3
starting_step = 14
number_of_training_steps = 65

#define sweep over the weight decay parameters
wds_postion_in_the_list = -5
wds = [0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125, 0.0015625, 0.00078125]
wds = wds[0:1]

#create parameter tuple
p=list(('tpsuvq', 'online', 't', 4, '1', '4096', 3, 0.25, 8, 'mse', 0.0001, wds[0], 'sub', starting_step, 14, 0.7))
param_list=[]
for w in wds:
  p[wds_postion_in_the_list]=w
  param_list.append(tuple(p))

print(param_list)

#load training data once for efficiency
fntmp = create_checkpoint_filename(param_list[0])
model, hyperparam = read_model(fntmp, True)
test_Loader = get_test_dataset(hyperparam, 4) # get testing data loader

#sequential evaluation loop
for step in range(number_of_training_steps):
  param_list_new=[]
  
  for ptmp in param_list:
    # read and evaluate model
    fntmp = create_checkpoint_filename(ptmp)
    model, hyperparam = read_model(fntmp, True) # get model
    y_pred, y = my_eval_model(model, test_Loader)

    # compute skill
    skill = compute_skill(y, y_pred)
    fnout = os.path.join('npys','expvar_'+os.path.split(fntmp)[-1])
    np.save(fnout, skill)

    #increment parameter tuple
    param_list_new.append( increment_parameter_tuple(ptmp) )
  param_list=param_list_new
  print(param_list)

# done with loop


