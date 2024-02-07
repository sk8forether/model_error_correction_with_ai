#!/scratch1/NCEPDEV/global/Tse-chun.Chen/anaconda3/envs/ltn/bin/python
# salloc --account=gsienkf --partition=bigmem --time=00:59:00 --nodes=1 --ntasks=40
import torch
from training import create_checkpoint_filename
from check_model import read_model, get_test_dataset, my_eval_model, compute_skill
import numpy as np
import os, math, time, importlib
from test_train_valid_splits import test_train_valid_splits
from training import Dataset_np as Dataset
from torch.utils.data import DataLoader

import training as t
importlib.reload(t)

torch.set_num_threads(int(os.cpu_count()/2))

SAVE_TRUTH=True

def my_eval_model(model, test_Loader):
  with torch.set_grad_enabled(False):
    model.eval()
    for X, y in test_Loader:
      y_pred = model(X)
  return y_pred, y

ptmp=['t', 4, '1', '4096', 3, 0.25, 32, 'mse', 0.0001, 1., 366,  365, 0.7]

#load training data once for efficiency
fntmp = create_checkpoint_filename(ptmp)
model, hyperparam = read_model(fntmp, True)

#load test data once for efficiency
print("loading data")
splits = test_train_valid_splits(hyperparam['testset'],
                hyperparam["end_of_training_day"], hyperparam["training_validation_length_days"])
test_slice = splits["test_slice"]
st = time.time()
test_set = t.Dataset_np(idx_include=test_slice, size='large', **hyperparam) # initiate dataset object
test_Loader = DataLoader(test_set, num_workers=0) # set up data loader
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


