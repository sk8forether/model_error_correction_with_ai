from check_model import model_to_nc
import shutil

fn = 'checks_batch/conv2d_t_4_1_4096_3_0.25_32_mse_0.0001_1.0_366_365_0.7'
model_to_nc(fn)
shutil.move('nn.nc',fn+'.nc')

fn = 'checks_batch/conv2d_q_4_1_4096_3_0.25_32_mse_0.0001_1.0_366_365_0.7'
model_to_nc(fn)
shutil.move('nn.nc',fn+'.nc')

fn = 'checks_batch/conv2d_u_4_1_4096_3_0.25_32_mse_0.0001_1.0_366_365_0.7'
model_to_nc(fn)
shutil.move('nn.nc',fn+'.nc')

fn = 'checks_batch/conv2d_v_4_1_4096_3_0.25_32_mse_0.0001_1.0_366_365_0.7'
model_to_nc(fn)
shutil.move('nn.nc',fn+'.nc')

