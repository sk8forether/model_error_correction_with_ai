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

ddd='/scratch2/BMC/gsienkf/Sergey.Frolov/fromStefan/npys_sergey3/ifs' # dir with normalizing mean/std
def denormal_out(x, vars_out):
    if vars_out == 'T':
        slice_out = slice(0,127)
    elif vars_out == 'U':
        slice_out = slice(127*1+1,127*2+1)
    elif vars_out == 'V':
        slice_out = slice(127*2+1,127*3+1)
    elif vars_out == 'Q':
        slice_out = slice(127*3+1,127*4+1)
        
    mean_out= torch.from_numpy(np.load(ddd+'_out_ranl_{}_mean_1d.npy'.format(trunc))[slice_out,None,None])
    std_out = torch.from_numpy(np.load(ddd+'_out_ranl_{}_std_1d.npy'.format(trunc)) [slice_out,None,None])
    output_size = std_out.shape[0]
    logging.info('Channel out  size: {}'.format(output_size))
    
    return x*std_out + mean_out

def read_input(): 

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
        sfcs.append(file_s[var].values) 

    sfcs.append(lons_m[None,]) # added for consistency with preprocessor
    sfcs.append(lats_m[None,]) 
    sfcs.append(lons_sin[None,]) 
    sfcs.append(lons_cos[None,]) 
    sfcs.append(np.ones((1,nlat,nlon))*date_in[:,None,None]) 

    # Combine the inputs
    #nbc = len(sfc_vars) #21 
    slice_f06 = slice(0,509) # 509 = 4*127 + 1 (127=nlev, 4=num  3d variables and one sfc variable as input)
    slice_sfc = slice(509,526) 
    
    ins = torch.cat([torch.from_numpy(np.concatenate(vals_f, axis=0)[None,]), 
                     torch.from_numpy(np.concatenate(sfcs, axis=0)[None,])], 1).float()
    print(np.shape(torch.from_numpy(np.concatenate(vals_f, axis=0)[None,])))
    print(np.shape(torch.from_numpy(np.concatenate(sfcs, axis=0)[None,])))
    # Prepare Normalizing mean and std
    mean_f06 = torch.from_numpy(np.load(ddd+'_f06_ranl_{}_mean_1d.npy'.format(trunc))[slice_f06]) # this now contains 3d and sfc variables. so, indices 0:509 are 3d as above, 7 sfc_vars, then 7 additional lat/lon/lon/date
    std_f06  = torch.from_numpy(np.load(ddd+'_f06_ranl_{}_std_1d.npy'.format(trunc)) [slice_f06])
    mean_sfc = torch.from_numpy(np.load(ddd+'_f06_ranl_{}_mean_1d.npy'.format(trunc))[slice_sfc])
    std_sfc  = torch.from_numpy(np.load(ddd+'_f06_ranl_{}_std_1d.npy'.format(trunc)) [slice_sfc])
    
    mean_in = torch.cat([mean_f06, mean_sfc],dim=0)[:,None,None]
    std_in  = torch.cat([std_f06,  std_sfc], dim=0)[:,None,None]
    X = (ins - mean_in)/std_in 

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
    file_f = xr.open_dataset(sys.argv[5]) # path to background input 3d file; /scratch2/BMC/gsienkf/Sergey.Frolov/fromStefan/2019112006/sfg_2019112006_fhr06_control_sub 
    file_s = xr.open_dataset(sys.argv[6]) # path to background input sfc file; /scratch2/BMC/gsienkf/Sergey.Frolov/fromStefan/2019112006/bfg_2019112006_fhr06_control_sub
if len(sys.argv) > 7:
    outd   = sys.argv[7] # output directory to write predicted increment; ; /scratch2/BMC/gsienkf/Laura.Slivinski/model_error_corr_work/
else:
    outd   = './'

date = pd.Timestamp('{}-{}-{}T{}'.format(indate[:4],indate[4:6],indate[6:8],indate[8:10]))

vars_in=['tmp','ugrd','vgrd','spfh','pressfc']
vars_out=['T_inc','sphum_inc','u_inc','v_inc','delz_inc','delp_inc','o3mr_inc']
sfc_vars=['csdlf','csdsf','csulf','csulftoa','csusf','csusftoa','land'] #7

if mode == 'T-only':
    vars_pred = ['T']
elif mode == 'TQ-only':
    vars_pred = ['T','Q']
elif mode == 'TQUV':
    vars_pred = ['T','Q','U','V']
else:
    logging.error("input mode {} not supported".format(mode))

#out_inc = "%s/fv3_increment6_predicted.nc"%outd # this is where the predicted increment will be saved
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
    file_i = xr.open_dataset('%s/%s/fv3_increment6.nc'%(outd,indate))
    logging.info("saving to %s/fv3_increment6_predicted.nc"%outd)
    
    y_pred = y_pred + [zeros]*(7-len(y_pred))

    for var,val in zip(vars_out, y_pred):
        if method in ['column',]:
            file_i[var].values = val[:,::-1]
        else:
            file_i[var].values = val
    file_i.to_netcdf("%s/fv3_increment6_predicted.nc"%outd, format='NETCDF4', engine='netcdf4') # make this consistent with above for local save location
    
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
