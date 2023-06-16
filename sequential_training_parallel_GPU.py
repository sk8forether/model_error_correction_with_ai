#!/scratch1/NCEPDEV/global/Tse-chun.Chen/anaconda3/envs/ltn/bin/python
import torch
import training as t
import time
#import importlib, training

#define sequential training configuration
end_day_postion_in_the_list=-3
training_length_position_in_the_list=-2
training_step_length=7
starting_step = 14
number_of_training_steps = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ptmp=[device,'tpsuvq', 'online', 't', 4, '1', '4096', 3, 0.25, 8, 'wnew', 0.0001, 1e-5, 'sub', 
            starting_step, training_step_length, 0.7]
print(ptmp)

#sequential traiing loop
for step in range(number_of_training_steps):
  #train
  st = time.time()
  t._train_(*ptmp)
  et = time.time()
  print('Execution time:', et-st, 'seconds')

  #update training length
  fn_before = t.create_checkpoint_filename(ptmp[1:])
  ptmp[end_day_postion_in_the_list] = ptmp[end_day_postion_in_the_list] + training_step_length
  ptmp[training_length_position_in_the_list] = ptmp[training_length_position_in_the_list] + training_step_length
  fn_after = t.create_checkpoint_filename(ptmp[1:])
  print('rename {} -> {}'.format(fn_before, fn_after))

  #reset network for next iteration
  t.reset_network(fn_before, fn_after)


#end loop


