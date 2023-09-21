#!/scratch1/NCEPDEV/global/Tse-chun.Chen/anaconda3/envs/ltn/bin/python
# salloc --account=gsienkf --partition=bigmem --time=00:59:00 --nodes=1 --ntasks=40
import torch
from training import create_checkpoint_filename
from check_model import read_model, get_test_dataset
import numpy as np
import os

torch.set_num_threads(40)

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


#define sweep over the weight decay parameters
wds_postion_in_the_list=-4
#number_of_wds=8
#wds=[0.01]*number_of_wds
#for i in range(1,number_of_wds):
#  wds[i]=wds[i-1]/10.0
wds=[1]

#create parameter traingin tuple
p=list(('t', 4, '1', '4096', 3, 0.25, 8, 'mse', 0.0001, 1., 366,  365, 0.7))
param_list=[]
for i in range(len(wds)):
  p[wds_postion_in_the_list]=wds[i]
  param_list.append(tuple(p))

print(param_list)

#load training data once for efficiency
fntmp = create_checkpoint_filename(param_list[0])
model, hyperparam = read_model(fntmp, True)

#load test data once for efficiency
print("loading data")
splits = test_train_valid_splits(hyperparam['testset'],
                hyperparam["end_of_training_day"], hyperparam["training_validation_length_days"])
test_slice = splits["test_slice"]
test_set = Dataset(idx_include=test_slice, **hyperparam) # initiate dataset object
batch_size=math.ceil(test_set.out.shape[0]/8)
test_Loader = DataLoader(test_set, batch_size=batch_size,num_workers=1) # set up data loader

#evaluation loop
for ptmp in param_list:
    # read and evaluate model
    fntmp = create_checkpoint_filename(ptmp)
    model, hyperparam = read_model(fntmp, True) # get model
    y_pred, y = my_eval_model(model, test_Loader)

    # compute skill
    skill = compute_skill(y, y_pred)
    fnout = os.path.join('npys','expvar_'+os.path.split(fntmp)[-1])
    np.save(fnout, skill)

# done with loop


