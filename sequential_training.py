#!/scratch1/NCEPDEV/global/Tse-chun.Chen/anaconda3/envs/ltn/bin/python
import torch
from training import Train_CONV2D , create_checkpoint_filename, reset_network
#mport importlib, training

#updates parameter tuple and resets the network
def my_reset(p_before):
  fn_before = create_checkpoint_filename(p_before)
  p_new=list(p_before)
  p_new[start_day_postion_in_the_list]=p_new[start_day_postion_in_the_list]+training_step_length
  p_new=tuple(p_new)
  fn_new = create_checkpoint_filename(p_new)
  reset_network(fn_before, fn_new)
  return p_new

#define sequential training configuration
training_step_length=7
start_day_postion_in_the_list=-3
starting_step = 406
number_of_training_steps = 8

#define sweep over the weight decay parameters
wds_postion_in_the_list=-5
number_of_wds=8
wds=[0.1]*number_of_wds
for i in range(1,number_of_wds):
  wds[i]=wds[i-1]/2.0

#create parameter traingin tuple
p=list(('tpsuvq', 'online', 't', 4, '1', '4096', 3, 0.25, 8, 'mse', 0.0001, wds[0], 'sub', starting_step, 14, 0.7))
param_list=[]
for i in range(len(wds)):
  p[wds_postion_in_the_list]=wds[i]
  param_list.append(tuple(p))
print(param_list)

#sequential traiing loop
for step in range(number_of_training_steps):
  #this bumper is needed for parallel jobs nested under Train_CONV2D
  if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    Train_CONV2D(param_list)
   
    param_list_new=[]
    for ptmp in param_list:
      param_list_new.append( my_reset(ptmp) )
    param_list=param_list_new
    print(param_list)

#end loop

#p=param_list[0]
#training._train_(0,p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12],p[13],p[14],p[15])
#Train_CONV2D(param_list)


