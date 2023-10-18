from time import time
import os, sys, types

num_threads = int(os.cpu_count()/4)
os.environ["MKL_NUM_THREADS"] = str(num_threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)
os.environ["OMP_NUM_THREADS"] = str(num_threads)
import numpy as np
import pandas

from pandas import date_range, Timedelta, Timestamp
import netCDF4
import xarray as xr
import datetime as datetime

import logging
logging.basicConfig(level=logging.INFO)

t0 = time()

pd = types.SimpleNamespace()
pd.date_range = date_range
pd.Timedelta = Timedelta
pd.Timestamp = Timestamp

vars_in=['tmp','ugrd','vgrd','spfh','pressfc']
vars_out=['T_inc','sphum_inc','u_inc','v_inc','delz_inc','delp_inc','o3mr_inc']
#vars_out_dict = {'t':'T_inc','q':'sphum_inc','u':'u_inc','v':'v_inc'}
sfc_vars=['csdlf','csdsf','csulf','csulftoa','csusf','csusftoa','land'] #7

num_predvar = 1 # hardcoding this for now; one variable is predicted (to use when reshaping y_pred and creating output file with correct number of output vars)

# csdlf     -- Clear Sky Downward Long Wave Flux
# csdsf     -- Clear Sky Downward Short Wave Flux
# csulf     -- Clear Sky Upward Long Wave Flux
# csulftoa  -- Clear Sky Upward Long Wave Flux at toa
# csusf     -- Clear Sky Upward Short Wave Flux
# csusftoa  -- Clear Sky Upward Short Wave Flux at toa
# land      -- sea-land-ice mask (0-sea, 1-land, 2-ice)

#fn_bfg="/scratch2/BMC/gsienkf/Sergey.Frolov/fromStefan/2019112006/sfg_2019112006_fhr06_control_sub"
#fn_sfg="/scratch2/BMC/gsienkf/Sergey.Frolov/fromStefan/2019112006/bfg_2019112006_fhr06_control_sub"
#inc_sfg = 'output'
#outd   = './'
#network_file = "/home/Sergey.Frolov/work/model_error/work/stefan_replay/checks/conv2d_t_4_1_4096_3_0.25_32_mse_0.0001_1.0_366_365_0.7.nc"

def relu(x):
    return np.maximum(x, [0],out=x)

def read_input(): 
    breakpoint()
    # time ## modified for consistency with preprocess.py
    date_j = date.to_julian_date()
    time_scales= [1, 365]
    time_sin = [np.sin(date_j*2*np.pi/period) for period in time_scales] #25,26
    time_cos = [np.cos(date_j*2*np.pi/period) for period in time_scales] #27,28
    time_h_m = [date.hour, date.month] #29,30 # raw hour and month info
    date_in = np.array(time_sin+time_cos+time_h_m, dtype=np.float32)
    
    # latlon
    lons_m, lats_m = file_f.lon.values, file_f.lat.values

    lons_sin = np.sin(lons_m*2*np.pi/360)
    lons_cos = np.cos(lons_m*2*np.pi/360)

    nlat, nlon = lons_m.shape

    # FCST input
    X = []
    for var in vars_in:
        val_f = file_f[var].values[0]

        if (var == 'pressfc'):
            val_f = np.log(val_f)[None]

        X.append(val_f)

    # SFC input
    for var in sfc_vars:
        X.append(file_s[var].values) 

    X.append(lons_m[None,]) # added for consistency with preprocessor
    X.append(lats_m[None,]) 
    X.append(lons_sin[None,]) 
    X.append(lons_cos[None,]) 
    X.append(np.ones((1,nlat,nlon))*date_in[:,None,None]) 

    X_ = np.concatenate(X, axis=0)

    logging.info('Input size: {}'.format(X_.shape))
    return X_

def forward(X):

    y_pred = relu(np.matmul(model.w0.T.to_numpy(),X.reshape((X.shape[0], X.shape[2]*X.shape[1])))+model.b0.to_numpy()[:, None])
    y_pred = relu(np.matmul(model.w1.T.to_numpy(),y_pred)+model.b1.to_numpy()[:, None])
    y_pred = np.matmul(model.w2.T.to_numpy(),y_pred)+model.b2.to_numpy()[:, None]
    y_pred = np.reshape(y_pred,(num_predvar,y_pred.shape[0],X.shape[1],X.shape[2]))

    return y_pred

def write_output(y_pred):
    global file_f

    # Save to files
    zeros = np.zeros(y_pred[0].shape,dtype=np.float32)

    # var_out = model.var_out

    if inc_sfg == 'output':
        breakpoint()

        logging.info("saving to "+out_file)

        y_pred = y_pred + [zeros]*(7-len(y_pred)) 
        
        file_i = xr.Dataset(data_vars = dict([(vars_out[0],(["lev","lat","lon"],zeros))]), coords=dict([('lon',(["lon"],file_f.grid_xt.data)),('lat',(["lat"],file_f.grid_yt.data)),('lev',(["lev"],range(1,128)))]))
        for i in range(1,len(vars_out)):
           file_i=file_i.assign(dict([(vars_out[i],(["lev","lat","lon"],zeros))]))

        for var,val in zip(vars_out, y_pred):
           file_i[var].values = val[:,::-1]
        file_i.to_netcdf(out_file, format='NETCDF4', engine='netcdf4') # make this consistent with above for local save location

    elif inc_sfg == 'update':
        logging.info("saving back to input file: fhr06_control")
        file_f.close()
        file_f = netCDF4.Dataset(out_file,'r+')

        for var,val in zip(['tmp','spfh','ugrd','vgrd',][:len(y_pred)],y_pred):
            file_f[var][:] += val
        file_f.close()

    else:
        logging.info("input inc_sfg {} not supported".format(inc_sfg))



## MAIN START ##
if __name__ == "__main__":
    logging.info('Inputs: %s', sys.argv)

    ### -- parse input
    network_file  = sys.argv[1] # neural network file location 
    fn_sfg = sys.argv[2]  #path to background input 3d file
    fn_bfg = sys.argv[3]  #path to background input 2d file
    if len(sys.argv) > 4:
        out_file = sys.argv[4]
        inc_sfg = 'update'
    else:
        inc_sfg = 'output'
        out_file = "./fv3_increment6_predicted.nc" # if we want this to be user-provided, a few mods are required
    print(inc_sfg)
    ### -- open files
    file_f = xr.open_dataset(fn_sfg)
    file_s = xr.open_dataset(fn_bfg)
    date = pandas.to_datetime(file_f.time)[0]
    indate = date.strftime('%Y%m%d%H')

    model = xr.load_dataset(network_file)

    write_output(forward(read_input()))


