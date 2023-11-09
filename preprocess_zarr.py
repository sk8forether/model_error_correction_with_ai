# Loading package
import itertools
import time
import sys

from joblib import Parallel, delayed
import pandas as pd
import xarray as xr
import numpy as np

# which components should be processed
ATM=True
NORMALIZE=True

dataDir='/scratch2/BMC/gsienkf/Sergey.Frolov/fromStefan/'   # directory for input data
#npyDir=dataDir+'npys_sergey2/ifs'                            # output directory
npyDir=dataDir+'../../Peter.Vaillancourt/Development/zarrtest'                            # output directory

def preprocess():
    '''preprocess the replay dataset from the nc files of reduced dataset into xarray using zarr'''
    
    time_scales=[1,365]                                                   # include hour of the day and day of the year info
    vars_in=['tmp','ugrd','vgrd','spfh','pressfc']                        # input forecast variables
    vars_out=['tmp','ugrd','vgrd','spfh','pressfc']                        # output variables
    sfc_vars=['csdlf','csdsf','csulf','csulftoa','csusf','csusftoa','land'] # boundary condition variables from surface file

    dates = [d for d in pd.date_range('2018-01-01T00', '2019-12-31T18', freq='6H')]     # preprocess data range
    
    date_in=[] # container for time info
    for date in dates:
        date_j = date.to_julian_date() # convert to julian date format
        time_sin = [np.sin(date_j*2*np.pi/period) for period in time_scales] #25,26 # sine wave with frequencies in time_scales
        time_cos = [np.cos(date_j*2*np.pi/period) for period in time_scales] #27,28 # cosine wave with frequencies in time_scales
        time_h_m = [date.hour, date.month] #29,30 # raw hour and month info
        date_in.append(time_sin+time_cos+time_h_m)
    date_in = np.array(date_in, dtype=np.float32)

    # get latlon info from a sample data
    sample = xr.open_dataset(dataDir+'/2019122000/sfg_2019122000_fhr06_control_sub')
    lons_d = sample.grid_xt
    lats_d = sample.grid_yt
    lons_m, lats_m = np.meshgrid(lons_d, lats_d) # get raw gridded lat and lon
    lons_sin = np.sin(lons_m*2*np.pi/360) # get sine of lon so that it is continueous between 0 and 360
    lons_cos = np.cos(lons_m*2*np.pi/360) # get cosine of lon so that it is continueous between 0 and 360
    nlon   = len(lons_d)
    nlat   = len(lats_d)
    ndates = len(dates)
    nlevs  = len(sample.pfull)
    
    sfc_size    = len(sfc_vars)+2+2+6    # lon, lat, lon_sin, lon_cos, time6
    varin_size  = nlevs*4+1+sfc_size     
    varout_size = nlevs*4+1 
    
    print(f"nlat={nlat}, nlon={nlon}")
    print("sfc size: {}, var in size: {}, var out size: {}, date size: {}".format(sfc_size, varin_size, varout_size, ndates))

    # prepare numpy array files in memory-map mode cause the full data is too large to live in memory.
    if ATM:
        # atmospheric forecast
        f06_sub  = np.lib.format.open_memmap(npyDir+'_f06_ranl_sub', mode='w+',shape=(ndates,varin_size, nlat,nlon), dtype=np.float32)
        if NORMALIZE:
          f06_sub_mean = np.load(npyDir+'_f06_ranl_sub_mean_1d.npy')
          f06_sub_std = np.load(npyDir+'_f06_ranl_sub_std_1d.npy')
        else:
          f06_sub_mean = np.zeros((varin_size,), dtype=np.float32)
          f06_sub_std  = np.ones((varin_size,), dtype=np.float32)

        # increment
        inc_sub  = np.lib.format.open_memmap(npyDir+'_out_ranl_sub', mode='w+',shape=(ndates,varout_size,nlat,nlon), dtype=np.float32)
        if NORMALIZE:
          inc_sub_mean = np.load(npyDir+'_out_ranl_sub_mean_1d.npy')
          inc_sub_std = np.load(npyDir+'_out_ranl_sub_std_1d.npy')
        else:
          inc_sub_mean = np.zeros((varout_size,), dtype=np.float32)
          inc_sub_std  = np.ones((varout_size,), dtype=np.float32)

    def write(index_d, date):
        '''writing data into the numpy array files'''
        print(date)
        YYYYMMDDHH = date.strftime('%Y%m%d%H') # current datetime
        PYYYYMMDDHH = (date + pd.Timedelta('6H')).strftime('%Y%m%d%H') # current datetime + 6h
        for suf, f06, inc in zip(['sub'],[f06_sub],[inc_sub]): # loop through sub (subsampled)
                vals_f = []
                vals_i = []
                file_f = xr.open_dataset('{}/{}/sfg_{}_fhr06_control_{}'.format(dataDir,YYYYMMDDHH,YYYYMMDDHH,suf)) # read forecast file
                file_a = xr.open_dataset('{}/{}/sfg_{}_fhr00_control_{}'.format(dataDir,PYYYYMMDDHH,PYYYYMMDDHH,suf)) # read analysis file
                file = xr.open_dataset('{}/{}/bfg_{}_fhr06_control_{}'.format(dataDir,YYYYMMDDHH,YYYYMMDDHH,suf)) # read boundary condition from file

                for var in vars_in: # loop through the variables
                    val_f = file_f[var].values[0]

                    if np.in1d(var,vars_out)[0]:
                        val_i = file_a[var].values[0] - val_f # compute increment (a-f)
                        if (var == 'pressfc'):
                            val_i = val_i[None] # add nex axis for surface pressure data to align with other variables
                        vals_i.append(val_i)

                    if (var == 'pressfc'):
                        val_f = np.log(val_f)[None] # compute log(Ps) for forecast and add new axis
                        
                    if (var == 'dpres'):
                        # compute cummulative pressure from surface for each level
                        val_f = np.cumsum(np.concatenate([val_f, np.zeros((1,nlat,nlon))],axis=0)[::-1],axis=1)[::-1]
                        val_f = -(val_f[1:] + val_f[:-1])/2 + file_f.pressfc.values # add surface pressure to get full pressure for each level

                    vals_f.append(val_f)
                for var in sfc_vars:
                    vals_f.append(file[var].values) # shape(1,32,64) # collect all variables in sfc_vars
                vals_f.append(lons_m[None,]) #21 # raw lon
                vals_f.append(lats_m[None,]) #22 # raw lat
                vals_f.append(lons_sin[None,]) #23 # sine of lon
                vals_f.append(lons_cos[None,]) #24 # cosine of lon
                vals_f.append(np.ones((1,nlat,nlon))*date_in[index_d][:,None,None]) # time information

                f06[index_d] = (np.concatenate(vals_f, axis=0) - f06_sub_mean[:,None,None])/f06_sub_std[:,None,None] # stack up all variables for each date
                inc[index_d] = (np.concatenate(vals_i, axis=0) - inc_sub_mean[:,None,None])/inc_sub_std[:,None,None] # stack up all variables for each date

            
    Parallel(n_jobs=40,verbose=0)(delayed(write)(i,d) for i,d in enumerate(dates)) # run write in threadded parallel
#   write(0,dates[0])
#   for i,d in enumerate(dates):write(i,d)

######################################################################################
preprocess()      # run the main program
   
