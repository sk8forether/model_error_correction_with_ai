#!/scratch1/NCEPDEV/global/Tse-chun.Chen/anaconda3/envs/ltn/bin/python
import torch
import training as t
import importlib
import time
importlib.reload(t)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ptmp=[device,'tpsuvq', 'online', 't', 4, '1', '4096', 3, 0.25, 8, 'mse', 0.0001, 1., 'sub', 375, 3, 0.7]
st = time.time()
t._train_(*ptmp)
et = time.time()
print('Execution time:', et-st, 'seconds')



