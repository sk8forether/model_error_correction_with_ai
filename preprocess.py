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
SFC=True

def preprocess():
    '''preprocess the replay dataset from the nc files of reduced dataset into numpy arrays'''
    
    time_scales=[1,365]                                                                 # include hour of the day and day of the year info
    vars_in=['tmp','pressfc','ugrd','vgrd','spfh',
             'dpres','dzdt','clwmr','rwmr','snmr','icmr','o3mr',]                       # input forecast variables
    vars_out=['tmp','pressfc','ugrd','vgrd','spfh',
              'dpres','dzdt','o3mr']                                                    # output variables
    sfc_vars=['acond','evcw_ave','evbs_ave','sbsno_ave','snohf','snowc_ave',
              'ssrun_acc','trans_ave','tmpsfc','tisfc','spfh2m','pevpr_ave','sfcr',
              'albdo_ave','csdlf','csdsf','csulf','csulftoa','csusf','csusftoa','land'] # 21 boundary condition variables from surface file
    
    ddd='./npys/ifs'

    dates = [d for d in pd.date_range('2019-12-20T00', '2019-12-20T12', freq='6H')]     # preprocess data range
    
    date_in=[] # container for time info
    for date in dates:
        date_j = date.to_julian_date() # convert to julian date format
        time_sin = [np.sin(date_j*2*np.pi/period) for period in time_scales] #25,26 # sine wave with frequencies in time_scales
        time_cos = [np.cos(date_j*2*np.pi/period) for period in time_scales] #27,28 # cosine wave with frequencies in time_scales
        time_h_m = [date.hour, date.month] #29,30 # raw hour and month info
        date_in.append(time_sin+time_cos+time_h_m)
    date_in = np.array(date_in, dtype=np.float32)

    # get latlon info from a sample data
    sample = xr.open_dataset('./sample_nc/2019122000/sfg_2019122000_fhr06_control_low')
    lons_d = sample.lon
    lats_d = sample.lat
    lons_m, lats_m = np.meshgrid(lons_d, lats_d) # get raw gridded lat and lon
    lons_sin = np.sin(lons_m*2*np.pi/360) # get sine of lon so that it is continueous between 0 and 360
    lons_cos = np.cos(lons_m*2*np.pi/360) # get cosine of lon so that it is continueous between 0 and 360
    nlon   = len(lons_d)
    nlat   = len(lats_d)
    ndates = len(dates)
    
    sfc_size    = len(sfc_vars)+2+2+6    # lon, lat, lon_sin, lon_cos, time6
    varin_size  = 1398                   # 509 + 7*127
    varout_size = 7*127 +1
    
    print(f"nlat={nlat}, nlon={nlon}")
    print("sfc size: {}, var in size: {}, var out size: {}, date size: {}".format(sfc_size, varin_size, varout_size, ndates))

    # prepare numpy array files in memory-map mode cause the full data is too large to live in memory.
    if SFC:
        # boundary conditions
        sfc_low  = np.lib.format.open_memmap(ddd+'_sfc_ranl_low', mode='w+',shape=(ndates,sfc_size,nlat,nlon), dtype=np.float32)
        sfc_sub  = np.lib.format.open_memmap(ddd+'_sfc_ranl_sub', mode='w+',shape=(ndates,sfc_size,nlat,nlon), dtype=np.float32)
        
    if ATM:
        # atmospheric forecast
        f06_low  = np.lib.format.open_memmap(ddd+'_f06_ranl_low', mode='w+',shape=(ndates,varin_size, nlat,nlon), dtype=np.float32)
        f06_sub  = np.lib.format.open_memmap(ddd+'_f06_ranl_sub', mode='w+',shape=(ndates,varin_size, nlat,nlon), dtype=np.float32)
        # increment
        inc_low  = np.lib.format.open_memmap(ddd+'_out_ranl_low', mode='w+',shape=(ndates,varout_size,nlat,nlon), dtype=np.float32)
        inc_sub  = np.lib.format.open_memmap(ddd+'_out_ranl_sub', mode='w+',shape=(ndates,varout_size,nlat,nlon), dtype=np.float32)

    def write(index_d, date):
        '''writing data into the numpy array files'''
        print(date)
        YYYYMMDDHH = date.strftime('%Y%m%d%H') # current datetime
        PYYYYMMDDHH = (date + pd.Timedelta('6H')).strftime('%Y%m%d%H') # current datetime + 6h
        for suf, f06, inc, sfc in zip(['low','sub'],[f06_low,f06_sub],[inc_low,inc_sub],[sfc_low,sfc_sub]): # loop through sub (subsampled) and low (spectral trunc)
            if ATM:
                # ATM files
                vals_f = []
                vals_i = []
                file_f = xr.open_dataset('./sample_nc/{}/sfg_{}_fhr06_control_{}'.format(YYYYMMDDHH,YYYYMMDDHH,suf)) # read forecast file
                file_a = xr.open_dataset('./sample_nc/{}/sfg_{}_fhr00_control_{}'.format(PYYYYMMDDHH,PYYYYMMDDHH,suf)) # read analysis file

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
                        val_f = np.cumsum(np.concatenate([val_f, np.zeros((1,nlat,nlon))],axis=0)[::-1],axis=1)[::-1] # compute cummulative pressure from surface for each level
                        val_f = -(val_f[1:] + val_f[:-1])/2 + file_f.pressfc.values # add surface pressure to get full pressure for each level

                    vals_f.append(val_f)

                f06[index_d] = np.concatenate(vals_f, axis=0) # stack up all variables for each date
                inc[index_d] = np.concatenate(vals_i, axis=0) # stack up all variables for each date

            if SFC:
                # SFC file
                sfcs = []
                file = xr.open_dataset('./sample_nc/{}/bfg_{}_fhr06_control_{}'.format(YYYYMMDDHH,YYYYMMDDHH,suf)) # read boundary condition from file
                for var in sfc_vars:
                    sfcs.append(file[var].values) # shape(1,32,64) # collect all variables in sfc_vars

                sfcs.append(lons_m[None,]) #21 # raw lon
                sfcs.append(lats_m[None,]) #22 # raw lat
                sfcs.append(lons_sin[None,]) #23 # sine of lon
                sfcs.append(lons_cos[None,]) #24 # cosine of lon
                sfcs.append(np.ones((1,nlat,nlon))*date_in[index_d][:,None,None]) # time information

                sfc[index_d] = np.concatenate(sfcs, axis=0) # stack up all variables for each date
            
    Parallel(n_jobs=40,verbose=0)(delayed(write)(i,d) for i,d in enumerate(dates)) # run write in threadded parallel
    #write(0,dates[0])

######################################################################################
preprocess() # run the main program

    
