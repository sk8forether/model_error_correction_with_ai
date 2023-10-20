#!/scratch1/NCEPDEV/global/Tse-chun.Chen/anaconda3/envs/ltn/bin/python
# salloc --account=gsienkf --partition=bigmem --time=00:59:00 --nodes=1 --ntasks=40
import torch
from training import create_checkpoint_filename
from check_model import read_model, get_test_dataset
import numpy as np
import os, math, time
from test_train_valid_splits import test_train_valid_splits
from training import Dataset_np as Dataset
from torch.utils.data import DataLoader

import training as t
importlib.reload(t)

torch.set_num_threads(int(os.cpu_count()/2))

SAVE_TRUTH=False

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

ptmp=['t', 4, '1', '4096', 3, 0.25, 8, 'mse', 0.0001, 1., 366,  36, 0.7]

#load training data once for efficiency
fntmp = create_checkpoint_filename(ptmp)
model, hyperparam = read_model(fntmp, True)

#load test data once for efficiency
print("loading data")
splits = test_train_valid_splits(hyperparam['testset'],
                hyperparam["end_of_training_day"], hyperparam["training_validation_length_days"])
test_slice = splits["test_slice"]
st = time.time()
test_set = t.Dataset_np(idx_include=test_slice, **hyperparam) # initiate dataset object
batch_size=32
test_Loader = DataLoader(test_set, batch_size=batch_size,num_workers=0) # set up data loader
et = time.time()
print('Load data:', et-st, 'seconds')

# compute prediction
y_pred, y = my_eval_model(model, test_Loader)

#save truth
if SAVE_TRUTH:
  fnout = os.path.join('npys','y_'+ptmp[0]+'.npy')
  np.save(fnout, y)

# compute skill
skill = compute_skill(y, y_pred)
fnout = os.path.join('npys','expvar_'+os.path.split(fntmp)[-1])
np.save(fnout, skill)


