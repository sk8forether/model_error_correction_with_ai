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

def denormal_out(x, vars_out):
    if vars_out == 'T':
        slice_out = slice(0,127)
    elif vars_out == 'U':
        slice_out = slice(127*1+1,127*2+1)
    elif vars_out == 'V':
        slice_out = slice(127*2+1,127*3+1)
    elif vars_out == 'Q':
        slice_out = slice(127*3+1,127*4+1)
        
    ddd='/scratch2/NCEPDEV/stmp1/Tse-chun.Chen/anal_inc/npys/ifs'
    mean_out= torch.from_numpy(np.load(ddd+'_out_ranl_{}_mean_1d.npy'.format(trunc))[slice_out,None,None])
    std_out = torch.from_numpy(np.load(ddd+'_out_ranl_{}_std_1d.npy'.format(trunc)) [slice_out,None,None])
    output_size = std_out.shape[0]
    logging.info('Channel out  size: {}'.format(output_size))
    
    return x*std_out + mean_out

def read_input(): # needs clean up with the proprocessor and dataset

    # time
    date_j = date.to_julian_date()
    time_scales= [1, 365]
    time_sin = [np.sin(date_j*2*np.pi/period) for period in time_scales] #25,26
    time_cos = [np.cos(date_j*2*np.pi/period) for period in time_scales] #27,28
    date_in = np.array(time_sin+time_cos, dtype=np.float32)
    
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
        sfcs.append(file_s[var].values) # shape(1,32,64)

    sfcs.append(lats_m[None,]) #22
    sfcs.append(lons_sin[None,]) #23
    sfcs.append(lons_cos[None,]) #24
    sfcs.append(np.ones((1,nlat,nlon))*date_in[:,None,None])

    # Combine the inputs
    nbc = 21
    slice_f06 = slice(0,509)
    slice_sfc = [14,15,16,17,18,19,20]+list(range(nbc,nbc+7)) 
    #slice_sfc = list(range(0,nbc))+list(range(nbc+1,nbc+8))
    
    ins = torch.cat([torch.from_numpy(np.concatenate(vals_f, axis=0)[None,]), 
                     torch.from_numpy(np.concatenate(sfcs, axis=0)[None,slice_sfc])], 1).float()
    # Prepare Normalizing mean and std
    ddd='/scratch2/NCEPDEV/stmp1/Tse-chun.Chen/anal_inc/npys/ifs'
    mean_f06 = torch.from_numpy(np.load(ddd+'_f06_ranl_{}_mean_1d.npy'.format(trunc))[slice_f06])
    std_f06  = torch.from_numpy(np.load(ddd+'_f06_ranl_{}_std_1d.npy'.format(trunc)) [slice_f06])
    mean_sfc = torch.from_numpy(np.load(ddd+'_sfc_ranl_{}_mean_1d.npy'.format(trunc))[slice_sfc])
    std_sfc  = torch.from_numpy(np.load(ddd+'_sfc_ranl_{}_std_1d.npy'.format(trunc)) [slice_sfc])
    
    mean_in = torch.cat([mean_f06, mean_sfc],dim=0)[:,None,None]
    std_in  = torch.cat([std_f06,  std_sfc], dim=0)[:,None,None]
    X = (ins - mean_in)/std_in

    input_size = ins.shape[1]
    logging.info('Channel in  size: {}'.format(input_size))
    return X

def column(var):
    if var == 'T':
        check_file = "/scratch2/NCEPDEV/stmp1/Tse-chun.Chen/anal_inc/checks/low-res-config/conv2d_tpsuvq_online_t_0_1_4096_3_0.25_8_mse_0.0001_0.05_sub"
    elif var == 'Q':
        check_file = "/scratch2/NCEPDEV/stmp1/Tse-chun.Chen/anal_inc/checks/low-res-config/conv2d_tpsuvq_online_q_0_1_4096_3_0.25_8_mse_0.0001_0.25_sub"
    elif var == 'U':
        check_file = "/scratch2/NCEPDEV/stmp1/Tse-chun.Chen/anal_inc/checks/low-res-config/conv2d_tpsuvq_online_u_0_1_4096_3_0.25_8_mse_0.0001_0.05_sub"
    elif var == 'V':
        check_file = "/scratch2/NCEPDEV/stmp1/Tse-chun.Chen/anal_inc/checks/low-res-config/conv2d_tpsuvq_online_v_0_1_4096_3_0.25_8_mse_0.0001_0.01_sub"

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

vars_in=['tmp','pressfc','ugrd','vgrd','spfh',]
         #'dpres','dzdt','clwmr','rwmr','snmr','icmr','o3mr',]
vars_out=['T_inc','sphum_inc','u_inc','v_inc','delz_inc','delp_inc','o3mr_inc']
sfc_vars=['acond','evcw_ave','evbs_ave','sbsno_ave','snohf','snowc_ave',
          'ssrun_acc','trans_ave','tmpsfc','tisfc','spfh2m','pevpr_ave','sfcr',
          'albdo_ave','csdlf','csdsf','csulf','csulftoa','csusf','csusftoa','land'] #21

if mode == 'T-only':
    vars_pred = ['T']
elif mode == 'TQ-only':
    vars_pred = ['T','Q']
elif mode == 'TQUV':
    vars_pred = ['T','Q','U','V']
else:
    logging.error("input mode {} not supported".format(mode))

if method in ["lowres", "column"]:
    sys.path.insert(0, "/home/Tse-chun.Chen/anal_inc/low_res_config/")
    from check_model import read_model
    from training import get_time, get_latlon
    import torch

out_inc = "fv3_increment6.nc"
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
    y_pred = Parallel(n_jobs=4)(delayed(column)(var) for var in vars_pred)
    updatef = sys.argv[5]

else:
    logging.error("input method {} not supported".format(method))

logging.info("Bias compute took {}s".format(time()-t0))
t0 = time()

# Save to files
zeros = np.zeros(y_pred[0].shape,dtype=np.float32)

if inc_sfg == 'output':
    ## sample file:
    file_i = xr.open_dataset('/scratch1/NCEPDEV/stmp2/Tse-chun.Chen/anal_inc/IFS_replay/2020010100/control/INPUT/fv3_increment6.nc')
    logging.info("saving to fv3_increment6.nc")
    
    y_pred = y_pred + [zeros]*(7-len(y_pred))

    for var,val in zip(vars_out, y_pred):
        if method in ['column',]:
            file_i[var].values = val[:,::-1]
        else:
            file_i[var].values = val
    file_i.to_netcdf("fv3_increment6.nc", format='NETCDF4', engine='netcdf4')
    
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
