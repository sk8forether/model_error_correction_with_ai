#!/scratch1/NCEPDEV/global/Tse-chun.Chen/anaconda3/envs/ltn/bin/python
# salloc --account=gsienkf --partition=bigmem --time=00:59:00 --nodes=1 --ntasks=40
import torch
from training import create_checkpoint_filename, compute_skill
from check_model import read_model
import numpy as np
import os, math, time, copy
from test_train_valid_splits import test_train_valid_splits
from training import Dataset_np as Dataset
from torch.utils.data import DataLoader
#from torchmetrics.functional import mean_squared_error

torch.set_num_threads(int(os.cpu_count()/2))

SLIDING_WINDOW=True
SAVE_TRUTH=True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def my_eval_model(model, test_Loader, device):
  y_pred = torch.zeros(test_Loader.dataset.out.shape)
  y = torch.zeros(test_Loader.dataset.out.shape)
  bs = test_Loader.batch_size
  with torch.set_grad_enabled(False):
    model.eval()
    for batch_id, (X_, y_) in enumerate(test_Loader):
        X_.to(device)
        y_pred_ = model(X_)
        y[batch_id*bs:batch_id*bs+y_.shape[0],:,:,:]=copy.deepcopy(y_)
        y_pred[batch_id*bs:batch_id*bs+y_.shape[0],:,:,:]=copy.deepcopy(y_pred_)
  return y_pred, y

def my_read_model(ptmp):
    fntmp = create_checkpoint_filename(ptmp[1:])
    model, hyperparam = read_model(fntmp, True, device=device)
    model = torch.nn.DataParallel(model)
    model.to(device)
    return model, hyperparam, fntmp


end_day_postion_in_the_list=-3
training_length_position_in_the_list=-2
training_step_length=7
starting_step = 365
number_of_training_steps = 104


#ptmp=[device, 't', 4, '1', '4096', 3, 0.25, 32, 'wnew', 0.0001, 1e-5,
#            starting_step, training_step_length, 0.7]

ptmp=[device, 't', 4, '1', '4096', 3, 0.25, 32, 'wnew', 0.0001, 1e-5, 1016, 658, 0.7]
number_of_training_steps = number_of_training_steps - 94


#load model
print("loading model")
model, hyperparam, fntmp = my_read_model(ptmp)

#load test data once for efficiency
print("loading data")
splits = test_train_valid_splits(hyperparam['testset'], 
                hyperparam["end_of_training_day"], hyperparam["training_validation_length_days"])
test_slice = splits["test_slice"]
test_set = Dataset(idx_include=test_slice, size="large", **hyperparam) # initiate dataset object
batch_size=128
test_Loader = DataLoader(test_set, batch_size=batch_size,num_workers=0) # set up data loader

print("Computing forward model")
st = time.time()
for step in range(number_of_training_steps):
    model, hyperparam, fntmp = my_read_model(ptmp)
    print(fntmp)
    y_pred, y = my_eval_model(model, test_Loader, device)
    skill = compute_skill(y, y_pred)
    fnout = os.path.join('npys','skill_'+os.path.split(fntmp)[-1]+'.npy')
    np.save(fnout, skill)
    ptmp[end_day_postion_in_the_list] = ptmp[end_day_postion_in_the_list] + training_step_length
    if SLIDING_WINDOW:
       ptmp[training_length_position_in_the_list] = ptmp[training_length_position_in_the_list] + training_step_length

et = time.time()
print('Forward time:', et-st, 'seconds')

# check that y was assembled correctley
#mean_squared_error(y, test_Loader.dataset.out)


# save y
if SAVE_TRUTH:
  fnout = os.path.join('npys','y_'+ptmp[1]+'.npy')
  np.save(fnout, y)


