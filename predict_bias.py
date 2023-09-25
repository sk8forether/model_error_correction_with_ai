from time import time
t0 = time()

import logging
logging.basicConfig(level=logging.INFO)

import os
import sys
from joblib import Parallel, delayed

import types
pd = types.SimpleNamespace()
from pandas import date_range, Timedelta, Timestamp
pd.date_range = date_range
pd.Timedelta = Timedelta
pd.Timestamp = Timestamp
import netCDF4
import xarray as xr
import numpy as np

from check_model import read_model # should be in current directory
from training import get_time, get_latlon
import torch

torch.set_num_threads(int(os.cpu_count()/2))

def denormal_out(x, vars_out):
    if vars_out == 'T':
        slice_out = slice(0,127)
    elif vars_out == 'U':
        slice_out = slice(127*1+1,127*2+1)
    elif vars_out == 'V':
        slice_out = slice(127*2+1,127*3+1)
    elif vars_out == 'Q':
        slice_out = slice(127*3+1,127*4+1)
        
    ddd='/scratch2/BMC/gsienkf/Sergey.Frolov/fromStefan/npys_sergey2/ifs'
    mean_out= torch.from_numpy(np.load(ddd+'_out_ranl_{}_mean_1d.npy'.format(trunc))[slice_out,None,None])
    std_out = torch.from_numpy(np.load(ddd+'_out_ranl_{}_std_1d.npy'.format(trunc)) [slice_out,None,None])
    output_size = std_out.shape[0]
    logging.info('Channel out  size: {}'.format(output_size))
    
    return x*std_out + mean_out

def read_input(): # needs clean up with the proprocessor and dataset

    # time ## modified for consistency with preprocess.py
    date_j = date.to_julian_date()
    time_scales= [1, 365]
    time_sin = [np.sin(date_j*2*np.pi/period) for period in time_scales] #25,26
    time_cos = [np.cos(date_j*2*np.pi/period) for period in time_scales] #27,28
    time_h_m = [date.hour, date.month] #29,30 # raw hour and month info
    date_in = np.array(time_sin+time_cos+time_h_m, dtype=np.float32)
    
    # latlon
    if method == "lowres":
        lons_d = file_f.lon
        lats_d = file_f.lat
        lons_m, lats_m = np.meshgrid(lons_d, lats_d)
    elif method == "column":
        lons_m, lats_m = file_f.lon.values, file_f.lat.values

    lons_sin = np.sin(lons_m*2*np.pi/360)
    lons_cos = np.cos(lons_m*2*np.pi/360)

    nlat, nlon = lons_m.shape

    # FCST input
    vals_f = []
    for var in vars_in:
        val_f = file_f[var].values[0]

        if (var == 'pressfc'):
            val_f = np.log(val_f)[None]

        vals_f.append(val_f)

    # SFC input
    sfcs = []
    for var in sfc_vars:
        sfcs.append(file_s[var].values) # was shape(1,32,64); now (7, 1, 64, 128)

    sfcs.append(lons_m[None,]) # added this for consistency with preprocessor
    sfcs.append(lats_m[None,]) 
    sfcs.append(lons_sin[None,]) 
    sfcs.append(lons_cos[None,]) 
    sfcs.append(np.ones((1,nlat,nlon))*date_in[:,None,None]) #11 (but is size 6 in dim 10 after date adjustemnt)
    #print('sfcs shape =', np.shape(sfcs)) #(11,) 
    #print('sfcs shape =', len(sfcs)) #(11,) 
    #print('sfcs[0] shape =', np.shape(sfcs[0])) #(1,64,128)
    #print('sfcs[10] shape =', np.shape(sfcs[10])) #(6,64,128)
    #print('sfcs_cat shape = ', np.shape(np.concatenate(sfcs, axis=0))) #(16,64,128)  

    # Combine the inputs
    #nbc = len(sfc_vars) #21 
    slice_f06 = slice(0,509) # 509 = 4*127 + 1 (127=nlev, 4=num  3d variables and one sfc variable as input)
    slice_sfc = slice(509,526) # since f06 and sfc are concatenated into one file, use this to slide the larger file. means we start from index 509 and go to end? should be 509 + len(sfc_vars) + 4+6 = 526
    #slice_sfc = [14,15,16,17,18,19,20]+list(range(nbc,nbc+7)) # if we subtract 14, given that sfc_vars went from len 14 to 7? what is this slicing though? maybe used to index csdlf first but now it's already first. Took last 7 vars from sfc_vars list and then added (21:28), which would grab lats, lons_sin, lons_cos, and the dates? so actually should slice_sfc just be [0:14] now (7 sfc vars and 7 addition lat/lon/lon/date vars?)
    #slice_sfc = list(range(0,nbc))+list(range(nbc+1,nbc+8))
    
    ins = torch.cat([torch.from_numpy(np.concatenate(vals_f, axis=0)[None,]), 
                     torch.from_numpy(np.concatenate(sfcs, axis=0)[None,])], 1).float()
    print(np.shape(torch.from_numpy(np.concatenate(vals_f, axis=0)[None,])))
    print(np.shape(torch.from_numpy(np.concatenate(sfcs, axis=0)[None,])))
    # Prepare Normalizing mean and std
    ddd='/scratch2/BMC/gsienkf/Sergey.Frolov/fromStefan/npys_sergey2/ifs'
    mean_f06 = torch.from_numpy(np.load(ddd+'_f06_ranl_{}_mean_1d.npy'.format(trunc))[slice_f06]) # this now contains 3d and sfc variables. so, indices 0:509 are 3d as above, 7 sfc_vars, then 7 additional lat/lon/lon/date
    std_f06  = torch.from_numpy(np.load(ddd+'_f06_ranl_{}_std_1d.npy'.format(trunc)) [slice_f06])
    mean_sfc = torch.from_numpy(np.load(ddd+'_f06_ranl_{}_mean_1d.npy'.format(trunc))[slice_sfc])
    std_sfc  = torch.from_numpy(np.load(ddd+'_f06_ranl_{}_std_1d.npy'.format(trunc)) [slice_sfc])
    #mean_sfc = torch.from_numpy(np.load(ddd+'_sfc_ranl_{}_mean_1d.npy'.format(trunc))[slice_sfc])
    #std_sfc  = torch.from_numpy(np.load(ddd+'_sfc_ranl_{}_std_1d.npy'.format(trunc)) [slice_sfc])
    
    mean_in = torch.cat([mean_f06, mean_sfc],dim=0)[:,None,None]
    std_in  = torch.cat([std_f06,  std_sfc], dim=0)[:,None,None]
    X = (ins - mean_in)/std_in ##!!! need to check; do I want to keep the additional variables included in preprocess or index them out? probably keep?

    input_size = ins.shape[1]
    logging.info('Channel in  size: {}'.format(input_size))
    return X

def column(var):
    if var == 'T':
        check_file = "/home/Sergey.Frolov/work/model_error/work/stefan_replay/checks/conv2d_t_4_1_4096_3_0.25_32_mse_0.0001_1.0_366_365_0.7"
    elif var == 'Q':
        check_file = "/home/Sergey.Frolov/work/model_error/work/stefan_replay/checks/conv2d_q_4_1_4096_3_0.25_32_mse_0.0001_1.0_366_365_0.7"
    elif var == 'U':
        check_file = "/home/Sergey.Frolov/work/model_error/work/stefan_replay/checks/conv2d_u_4_1_4096_3_0.25_32_mse_0.0001_1.0_366_365_0.7"
    elif var == 'V':
        check_file = "/home/Sergey.Frolov/work/model_error/work/stefan_replay/checks/conv2d_v_4_1_4096_3_0.25_32_mse_0.0001_1.0_366_365_0.7"
    elif var == 'PS': # not fully implemented yet
        check_file = "/home/Sergey.Frolov/work/model_error/work/stefan_replay/checks/conv2d_ps_4_1_4096_3_0.25_32_mse_0.0001_1.0_366_365_0.7"

    model = read_model(check_file)

    # preprocess input
    X = read_input()

    # predict biases
    with torch.set_grad_enabled(False):
        model.eval()
        y_pred = denormal_out(model(X),var).numpy()[0]
        
    return y_pred

logging.info("Defs took {}s".format(time()-t0))
t0 = time()

## MAIN START ##
logging.info('Inputs: %s', sys.argv)

method  = sys.argv[1] # moveavg, lowres, column
inc_sfg = sys.argv[2] # output (for pure forecast), update (for 3dvar)
mode    = sys.argv[3] # T-only, TQ-only, TQUV
indate  = sys.argv[4] # 2019112006

if len(sys.argv) > 5:
    file_f = xr.open_dataset(sys.argv[5])
    file_s = xr.open_dataset(sys.argv[6])

date = pd.Timestamp('{}-{}-{}T{}'.format(indate[:4],indate[4:6],indate[6:8],indate[8:10]))

vars_in=['tmp','ugrd','vgrd','spfh','pressfc']
vars_out=['T_inc','sphum_inc','u_inc','v_inc','delz_inc','delp_inc','o3mr_inc']
sfc_vars=['csdlf','csdsf','csulf','csulftoa','csusf','csusftoa','land'] #7
#sfc_vars=['acond','evcw_ave','evbs_ave','sbsno_ave','snohf','snowc_ave',
#          'ssrun_acc','trans_ave','tmpsfc','tisfc','spfh2m','pevpr_ave','sfcr',
#          'albdo_ave','csdlf','csdsf','csulf','csulftoa','csusf','csusftoa','land'] #21

if mode == 'T-only':
    vars_pred = ['T']
elif mode == 'TQ-only':
    vars_pred = ['T','Q']
elif mode == 'TQUV':
    vars_pred = ['T','Q','U','V']
else:
    logging.error("input mode {} not supported".format(mode))

out_inc = "fv3_increment6.nc" # this is where it will get saved; add directory path if you don't want it in the CD
if method == "moveavg":
    y_pred = Parallel(n_jobs=4)(delayed(moveavg)(var) for var in vars_pred)
    updatef = sys.argv[5]
elif method == "lowres":
    trunc = 'low'
    y_pred_low = Parallel(n_jobs=4)(delayed(lowres)(var) for var in vars_pred)
    y_pred = spec_padding(y_pred_low)
    updatef = sys.argv[5][:-4]

elif method == "column":
    trunc = 'sub'
    #y_pred = Parallel(n_jobs=4)(delayed(column)(var) for var in vars_pred) #
    y_pred = Parallel(n_jobs=1)(delayed(column)(var) for var in vars_pred)
    updatef = sys.argv[5]

else:
    logging.error("input method {} not supported".format(method))

logging.info("Bias compute took {}s".format(time()-t0))
t0 = time()

# Save to files
zeros = np.zeros(y_pred[0].shape,dtype=np.float32)

if inc_sfg == 'output':
    ## sample file:
    file_i = xr.open_dataset('/scratch2/BMC/gsienkf/Laura.Slivinski/model_error_corr_data/%s/fv3_increment6.nc'%indate) #later, make this more general
    logging.info("saving to fv3_increment6.nc") # make this consistent with above for local save location
    
    y_pred = y_pred + [zeros]*(7-len(y_pred))

    for var,val in zip(vars_out, y_pred):
        if method in ['column',]:
            file_i[var].values = val[:,::-1]
        else:
            file_i[var].values = val
    file_i.to_netcdf("fv3_increment6.nc", format='NETCDF4', engine='netcdf4') # make this consistent with above for local save location
    
elif inc_sfg == 'update':
    logging.info("saving back to input file: fhr06_control")
    file_f.close()
    file_f = netCDF4.Dataset(updatef,'r+')
        
    for var,val in zip(['tmp','spfh','ugrd','vgrd',][:len(y_pred)],y_pred):
        if method in ['column',]:
            file_f[var][:] += val
        else:
            file_f[var][:] += val[:,::-1]
    file_f.close()

else:
    logging.info("input inc_sfg {} not supported".format(inc_sfg))

logging.info("Bias saving took {}s".format(time()-t0))
