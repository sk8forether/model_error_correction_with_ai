#!/scratch1/NCEPDEV/global/Tse-chun.Chen/anaconda3/envs/ltn/bin/python
import torch
from training import Train_CONV2D 
#import importlib, training

#define sweep over the weight decay parameters
wds_postion_in_the_list=-5
#number_of_wds=8
#wds=[0.01]*number_of_wds
#for i in range(1,number_of_wds):
#  wds[i]=wds[i-1]/10.0
wds=[1, 0.1]

#create parameter traingin tuple
p=list(('tpsuvq', 'online', 't', 4, '1', '4096', 3, 0.25, 8, 'mse', 0.0001, wds[0], 'sub', 375, 5, 0.7))
param_list=[]
for i in range(len(wds)):
  p[wds_postion_in_the_list]=wds[i]
  param_list.append(tuple(p))
print(param_list)

#sequential traiing loop
if __name__ == '__main__':
  torch.multiprocessing.set_start_method('spawn', force=True)
  Train_CONV2D(param_list)
pass

