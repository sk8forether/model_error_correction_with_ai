#!/scratch1/NCEPDEV/global/Tse-chun.Chen/anaconda3/envs/ltn/bin/python
import torch
from training import create_checkpoint_filename, compute_skill
from check_model import read_model, get_test_dataset
import numpy as np
import os, copy, time

#import importlib
#import check_model as cm

torch.set_num_threads(int(os.cpu_count()/2))

# evaluate model
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

#define sequential training configuration
end_day_postion_in_the_list=-3
training_length_position_in_the_list=-2
training_step_length=7
start_day_postion_in_the_list=-3
starting_step = 14
number_of_training_steps = 64

#define test params list and load the first model
ptmp=['tpsuvq', 'online', 't', 4, '1', '4096', 3, 0.25, 8, 'wnew', 0.0001, 1e-5, 'sub',
            starting_step, training_step_length, 0.7]
fntmp = create_checkpoint_filename(ptmp)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, hyperparam = read_model(fntmp, True, device=device)
del model

#load test data once for efficiency
test_Loader = get_test_dataset(hyperparam, num_workers=0, batch_size=8) # get testing data loader

#sequential evaluation loop
stAll = time.time()
for step in range(number_of_training_steps):
    st = time.time()

    # read and evaluate model
    fntmp = create_checkpoint_filename(ptmp)
    model, hyperparam = read_model(fntmp, True, device=device) # get model
    model = torch.nn.DataParallel(model)
    model.to(device)
    y_pred, y = my_eval_model(model, test_Loader, device)

    # compute skill
    skill = compute_skill(y, y_pred)
    fnout = os.path.join('npys','expvar_'+os.path.split(fntmp)[-1])
    print('Saving {}'.format(fnout))
    np.save(fnout, skill)

    #increment parameter tuple
    ptmp[end_day_postion_in_the_list] = ptmp[end_day_postion_in_the_list] + training_step_length
    ptmp[training_length_position_in_the_list] = ptmp[training_length_position_in_the_list] + training_step_length
    et = time.time()
    print('Ellapse time={} seconds'.format(et-st))

etAll = time.time()
print('Ellapse time All={} minutes'.format((etAll-stAll)/60))


