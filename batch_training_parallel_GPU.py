#!/scratch1/NCEPDEV/global/Tse-chun.Chen/anaconda3/envs/ltn/bin/python
import torch
import training as t
import importlib
import time
importlib.reload(t)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ptmp=[device, 't', 4, '1', '4096', 3, 0.25, 32, 'mse', 0.0001, 1., 366,  365, 0.7] #Done 
ptmp=[device, 'q', 4, '1', '4096', 3, 0.25, 32, 'mse', 0.0001, 1., 366,  365, 0.7] #Done 
ptmp=[device, 'u', 4, '1', '4096', 3, 0.25, 32, 'mse', 0.0001, 1., 366,  365, 0.7] #done
ptmp=[device, 'v', 4, '1', '4096', 3, 0.25, 32, 'mse', 0.0001, 1., 366,  365, 0.7] #retrain  now
ptmp=[device, 'ps', 4, '1', '4096', 3, 0.25, 32, 'mse', 0.0001, 1., 366,  365, 0.7] #done

st = time.time()
t._train_(*ptmp)
et = time.time()
print('Execution time:', et-st, 'seconds')



