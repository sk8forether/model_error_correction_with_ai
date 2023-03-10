# Loading package
import os
import itertools
import sys
import time
from copy import deepcopy
import logging
logging.basicConfig(level=logging.DEBUG)

import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.multiprocessing import Pool
from torch.utils import data
from torch.utils.data import Subset, DataLoader, SubsetRandomSampler
from test_train_valid_splits import test_train_valid_splits
import random

# Loading self-defined package
from model import CONV2D

# Define training functions and utilities
def Train_CONV2D(param_list):
    '''Get training setup from a list hyperparameters (for hyper search) and Distribute each training task to different GPUs (max:8).
    This additional layer is for: 
    * training each task on 1 gpu in parallel
    * each node runs 8 gpus
    * scheduler cannot access just a single gpu without occupying the other 7'''
    
    param_list = [(n,)+param for n,param in enumerate(param_list)] # assign GPU ID to each task
    
    with Pool(processes = len(param_list)) as p:
        p.starmap(_train_, param_list) # submit tasks to the pool of processes

def _train_(rank,
            vars_f06, vars_sfc, vars_out, testset, kernel_sizes, 
            channels, n_conv, p, bs, loss, lr, wd, trunc,
            end_of_training_day, training_validation_length_days, tv_ratio):
    '''Run individual training task. Called by Train_CONV2D'''
    
    params = locals() # get local variables i.e. the input parameters 
    logging.info("rank: {} {}".format(rank, params))
    naming = ''
    for k in [key for key,val in params.items()][1:]:
        naming += '_'+str(params[k]) # concat the input parameters into a string

    checkfile = './checks/conv2d'+naming # filename for training checkpoints
    logging.info("rank: {}, check file: {}".format(rank, checkfile))
    
    ######################################################################    
    # Train_Valid DATASET
    # define the training and validation index range (indp test range defined in check_model)
    splits = test_train_valid_splits(testset, end_of_training_day, training_validation_length_days)
    train_valid_slice = splits["train_valid_slice"]
    training_to_validation_ratio = tv_ratio

    logging.info('rank: {}, Generating train_valid_set'.format(rank))
    Dataset = Dataset_np
    # initialize dataset object containing training datasets
    train_valid_set = Dataset(idx_include=train_valid_slice,
                              vars_f06=vars_f06,vars_sfc=vars_sfc,vars_out=vars_out,
                              trunc=trunc,)
    
    input_size  = len(train_valid_set[0][0])
    output_size = len(train_valid_set[0][1])
    volumn_size = np.prod(train_valid_set[0][1].shape)
    logging.info("rank: {}, volumn size: {}, dataset length: {}".format(rank, volumn_size, len(train_valid_set)))
    
    # Set up data loader
    
    # split training and validation data range
    idx = np.arange(len(train_valid_set))
    random.shuffle(idx)
    train_inds = list(idx[0:round(len(train_valid_set)*training_to_validation_ratio)])
    valid_inds = list(idx[round(len(train_valid_set)*training_to_validation_ratio):])
    logging.info("rank: {}, train_set time size: {}".format(rank, len(train_inds)))
    logging.info("rank: {}, valid_set time size: {}".format(rank, len(valid_inds)))
    
    valid_sampler = SubsetRandomSampler(valid_inds) # sample from the defined validation index range
    train_sampler = SubsetRandomSampler(train_inds) # sample from the defined training index range
    
    # initialize data loaders
    valid_loader = DataLoader(train_valid_set, batch_size=bs, num_workers=0, sampler=valid_sampler,)
    train_loader = DataLoader(train_valid_set, batch_size=bs, num_workers=0, sampler=train_sampler,)
    
    ######################################################################    
    # MODEL
    logging.info('rank: {}, Setting up training'.format(rank))
    logging.info('rank: {}, setting up model'.format(rank))
    
    model = CONV2D(input_size, output_size, kernel_sizes, channels, n_conv, p,) # initialize model object
    model.to(rank) # send model to gpu

    logging.info('rank: {}, setting up loss'.format(rank))
    
    # define loss functions
    if loss == 'mse':
        criterion = torch.nn.MSELoss(reduction='sum')
    elif loss == 'mae':
        criterion = torch.nn.L1Loss(reduction='sum')
    else:
        logging.error('rank: {}, Loss not supported!!'.format(rank))
        exit()

    logging.info('rank: {}, setting up optimizer'.format(rank))
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=wd, amsgrad=False) # initialize optimizer
    
    if os.path.isfile(checkfile):
        # read in previous checkpoint file (if exists) to continue training after being interrupted.
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(checkfile,map_location=map_location)
            
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        best_model = deepcopy(model.state_dict())
        best_optim = deepcopy(optimizer.state_dict())
        epoch = checkpoint['epoch']
        epo_init = checkpoint['epoch']+1
        train_losses = checkpoint['train_loss'][:epo_init]
        valid_losses  = checkpoint['valid_loss'][:epo_init]
        max_time = checkpoint['max_time']
        impatience = checkpoint['impatience']

        logging.info('rank: {}, Continue training from previous model, Previous epoches: {}'.format(rank,epo_init))

    else:
        epo_init = 0       # initial epoch number 
        best_model = []    # current best model parameters
        best_optim = []    # current best optimizer states
        train_losses = []  # history of train loss
        valid_losses = []  # history of valid loss
        max_time = 0       # keep track of maximum time used for each epoch
        epoch = 0          # keep track of number of epoches

    max_epoches = 500 # maximum training epoches before forced termination
    patience = 20     # maximum number of consecutive epoches that does not decrease the valid loss (for early stopping)
    
    
    ######################################################################    
    # TRAINING
    try:# prevent raising OOM, which leaves no checkpoint file.

        logging.info('rank: {}, Start training'.format(rank))
        np.random.seed(1337) # ensure reproducibility. for the random sampling

        for epoch in range(epo_init, max_epoches): # loop through training epoches

            ###########
            ## Training
            train_loss = 0
            start_time = time.time()
            model.train() # set model in training mode

            for batch_id, (X, y) in enumerate(train_loader): # loop through mini batches of the training dataset
                batch_time = time.time()
                X, y = X.to(rank), y.to(rank) # send data to gpu
                optimizer.zero_grad()         # reset gradient
                y_pred = model(X)             # get output from model
                loss = criterion(y_pred, y)   # compute loss
                loss.backward()               # compute gradient
                optimizer.step()              # update parameter
                train_loss += loss.item()     # sum loss for all batches

            train_loss /= len(train_inds)*volumn_size # compute mean loss
            train_losses.append(train_loss)           # record training loss
            
            logging.info("rank: {}, Train epoch: {}, Loss: {:.4f}, Current Minimum: {:.4f} at epoch# {}".format(rank, epoch, train_loss, min(train_losses), np.argmin(train_losses)))

            #############
            ## Validation
            with torch.set_grad_enabled(False):   # disable gradient
                model.eval()                      # set model in evaluation mode
                valid_loss = 0
                for X, y in valid_loader:         # loop over every sample in validation set
                    X, y = X.to(rank), y.to(rank) # send data to gpu
                    y_pred = model(X)             # evaluate model
                    y_diff_s = (y_pred - y)**2    # compute loss
                    valid_loss += torch.sum(y_diff_s).item() # sum all loss
                    
            valid_loss /= 244*volumn_size    # average loss
            valid_losses.append(valid_loss)  # record validation loss

            logging.info("rank: {}, Valid epoch: {}, Loss: {:.4f}, Current Minimum: {:.4f} at epoch# {}".format(rank, epoch, valid_loss, min(valid_losses), np.argmin(valid_losses)))

            impatience = epoch - np.argmin(valid_losses) # check number of epoches from last validation loss minimum
            if min(valid_losses) == valid_loss: # if current epoch has the minimal valid loss
                best_model = deepcopy(model.state_dict()) # extract model parameter
                best_optim = deepcopy(optimizer.state_dict()) # extract optimizer state
                
                logging.info('rank: {}, Reach new min, Saving checkpoint \n'.format(rank))
                _checkpoint_(rank,checkfile,best_model,best_optim,epoch,train_losses,valid_losses,impatience,max_time) # save history and current state to checkpoint file

            check_gpu(rank) # check gpu status
            current_time = time.time()
            elapsed_time = current_time - start_time
            logging.info("rank: {}, Elapsed time: {:.1f} \n".format(rank,elapsed_time))

            max_time = int(max(max_time, elapsed_time)) # update maximum time for each epoch

            if impatience > patience: # break from training if there have been too many consecutive epoches that does not decrease the valid loss
                logging.info('rank: {}, Break for impatience and save model \n'.format(rank))
                break

    except RuntimeError as e:
        if 'out of memory' in str(e): #catches only OOM
            impatience = 999  # mark the impatience status so that it is different from regular impatience break.
            logging.info("rank: {}, Out of memory!! checking in the checkfile".format(rank))
        else: 
            raise e
            
    _checkpoint_(rank,checkfile,best_model,best_optim,epoch,train_losses,valid_losses,impatience,max_time) # save history and current state to checkpoint file

class Dataset_np(data.Dataset):
    '''Define and Preprocess input and output data'''
    def __init__(self, idx_include=slice(40,None), # skip first 40 samples
                       vars_f06='tpsuvq',
                       vars_sfc='subset-cyc',
                       vars_out='t',
                       trunc='low',
                       **kwargs):
        
        t = time.time()
        # slicing input forecast variables
        if vars_f06 == 'tpsuvq':
            slice_f06 = slice(0,509)
        elif vars_f06 == 'tpsuvqp':
            slice_f06 = slice(0,636)
        elif vars_f06 == 'all':
            slice_f06 = slice(0,1398)
        
        nbc = 21
        # slicing input boundary variables
        if vars_sfc == 'subset-alltl':
            slice_sfc = slice(None,None)
        elif vars_sfc == 'subset-cyc': # 509+21+7
            slice_sfc = list(range(0,nbc))+list(range(nbc+1,nbc+8))
        elif vars_sfc == 'cli': # 509+4
            slice_sfc = [nbc,nbc+1]+[nbc+8,nbc+9]
        elif vars_sfc == 'cyc': # 509+7 lats_m, lons_sin, lons_cos, day_sin, year_sin, day_cos, year_cos
            slice_sfc = list(range(nbc+1,nbc+8))
        elif vars_sfc == 'subset': # 509+21
            slice_sfc = slice(0,nbc)
        elif vars_sfc == 'online': # 'csdlf','csdsf','csulf','csulftoa','csusf','csusftoa','land' +7
            slice_sfc = [14,15,16,17,18,19,20]+list(range(nbc+1,nbc+8))
        
        # slicing output variables
        if vars_out == 't':
            slice_out = slice(0,127)
        elif vars_out == 'u':
            slice_out = slice(127*1+1,127*2+1)
        elif vars_out == 'v':
            slice_out = slice(127*2+1,127*3+1)
        elif vars_out == 'q':
            slice_out = slice(127*3+1,127*4+1)
        elif vars_out == 'p':
            slice_out = slice(127*4+1,127*5+1)
        elif vars_out == 'z':
            slice_out = slice(127*5+1,127*6+1)
        elif vars_out == 'oz':
            slice_out = slice(127*6+1,127*7+1)
        
        ddd='/scratch2/NCEPDEV/stmp1/Tse-chun.Chen/anal_inc/npys/ifs' # dataset location
            
        self.ins = []
        # load data in memory map mode (allows slicing without actually loading the data)
        # 4D dataset [batch_size, channels, height, width]
        f06_in = np.load(ddd+'_f06_ranl_'+trunc,mmap_mode='r')[idx_include,slice_f06]
        sfc_in = np.load(ddd+'_sfc_ranl_'+trunc,mmap_mode='r')[idx_include,slice_sfc]
        out    = np.load(ddd+'_out_ranl_'+trunc,mmap_mode='r')[idx_include,slice_out]
        
        self.ndates, _, self.nlat, self.nlon = f06_in.shape # get data shape
        
        # convert data from numpy to torch tensor
        self.ins = [torch.from_numpy(np.copy(f06_in)),
                    torch.from_numpy(np.copy(sfc_in))] 
        self.ins = torch.cat(self.ins,1)
        self.out = torch.from_numpy(np.copy(out))
        
        print('Channel in  size: {}'.format(self.ins.shape[1]))
        print('Channel out size: {}'.format(self.out.shape[1]))
        print('Time snapshots: {}'.format(self.ndates))
        
        # read precomputed mean and std for the input and output from numpy to torch tensor
        mean_f06 = torch.from_numpy(np.load(ddd+'_f06_ranl_{}_mean_1d.npy'.format(trunc))[slice_f06])
        std_f06  = torch.from_numpy(np.load(ddd+'_f06_ranl_{}_std_1d.npy'.format(trunc)) [slice_f06])
        mean_sfc = torch.from_numpy(np.load(ddd+'_sfc_ranl_{}_mean_1d.npy'.format(trunc))[slice_sfc])
        std_sfc  = torch.from_numpy(np.load(ddd+'_sfc_ranl_{}_std_1d.npy'.format(trunc)) [slice_sfc])
        self.mean_in = torch.cat([mean_f06, mean_sfc],dim=0)[:,None,None]
        self.std_in  = torch.cat([std_f06,  std_sfc], dim=0)[:,None,None]
        self.mean_out= torch.from_numpy(np.load(ddd+'_out_ranl_{}_mean_1d.npy'.format(trunc))[slice_out,None,None])
        self.std_out = torch.from_numpy(np.load(ddd+'_out_ranl_{}_std_1d.npy'.format(trunc)) [slice_out,None,None])
            
        self.ins = self.__normal_in__(self.ins)
        self.out = self.__normal_out__(self.out)
        print('time preparing data: {}s'.format(time.time()-t))
        
    def __normal_in__(self,x):
        '''normalize input data'''
        return (x - self.mean_in)/self.std_in
    def __normal_out__(self,x):
        '''normalize output data'''
        return (x - self.mean_out)/self.std_out
    def __len__(self):
        '''get length of the training dataset'''
        return self.ndates
    def __getitem__(self, index):
        '''define index in the training dataset for dataloader'''
        return self.ins[index], self.out[index]

## UTILITIES

def _checkpoint_(rank,checkfile,best_model,best_optim,epoch,train_losses,valid_losses,impatience,max_time):
    '''save intermediate training process for the checkpoint file'''
    torch.save({'model_state_dict': best_model,
                'optimizer_state_dict': best_optim,
                'epoch': epoch,
                'train_loss': train_losses,
                'valid_loss': valid_losses,
                'impatience': impatience,
                'max_time': max_time}, checkfile)
    logging.info('rank: {}, check file: {}'.format(rank,checkfile))

def check_gpu(rank):
    '''print memory usage of a gpu. bug in 1.7 segmentation fault when nothing was put in gpu'''
    logging.info('rank: {}: Allocated: {} GB'.format(rank,round(torch.cuda.memory_allocated(rank)/1024**3,1),))
    logging.info('rank: {}: Cached:    {} GB'.format(rank,round(torch.cuda.memory_reserved(rank)/1024**3,1),))

def get_slice(vars_f06,vars_sfc,vars_out):
    base_f_list = ['tmp','ugrd','vgrd','spfh','pressfc','dpres','dzdt','hgtsfc',
                   'clwmr','dzdt','grle','icmr','o3mr','rwmr','snmr',]
    if vars_f06 == 'tpsuvq':
        slice_f06 = base_f_list[:5]
    elif vars_f06 == 'tpsuvqp':
        slice_f06 = base_f_list[:6]
    elif vars_f06 == 'all':
        slice_f06 = base_f_list

    #nbc = 21
    base_s_list = ['acond','evcw_ave','evbs_ave','sbsno_ave','snohf','snowc_ave',
                   'ssrun_acc','trans_ave','tmpsfc','tisfc','spfh2m','pevpr_ave','sfcr',
                   'albdo_ave','csdlf','csdsf','csulf','csulftoa','csusf','csusftoa','land']
    if vars_sfc == 'subset-alltl':
        slice_sfc    = base_s_list
        slice_time   = slice(None,None)
        slice_latlon = slice(None,None)
    elif vars_sfc == 'subset-cyc':
        slice_sfc    = base_s_list
        slice_time   = slice(0,4)
        slice_latlon = slice(2,None)
    elif vars_sfc == 'cli':
        slice_sfc    = []
        slice_time   = slice(4,None)
        slice_latlon = slice(0,2)
    elif vars_sfc == 'cli-cyc':
        slice_sfc    = []
        slice_time   = slice(0,4)
        slice_latlon = slice(2,None)
    elif vars_sfc == 'subset':
        slice_sfc    = base_s_list
        slice_time   = slice(0,0)
        slice_latlon = slice(0,0)
    elif vars_sfc == 'all':
        slice_sfc    = base_s_list
        slice_time   = slice(0,0)
        slice_latlon = slice(0,0)
        
    if vars_out == 't':
        slice_out = ['tmp']
    elif vars_out == 'u':
        slice_out = ['ugrd']
    elif vars_out == 'v':
        slice_out = ['vgrd']
    elif vars_out == 'q':
        slice_out = ['spfh']
    elif vars_out == 'ps':
        slice_out = ['pressfc']
        logging.warning("pressfc will result in nan because of the missing value in the inc_std file")
        
    return slice_f06, slice_sfc, slice_time, slice_latlon, slice_out

def get_grids(sel_type=5):
    if sel_type is None:
        return slice(None),slice(None)
    elif isinstance(sel_type, int):
        itvl = int(384*sel_type/100)
        i = np.random.randint(itvl)
        return slice(i,None,itvl),slice(i,None,itvl)
        #return np.random.choice(384,size=int(384*sel_type/100)), np.random.choice(768,size=int(768*sel_type/100))
    elif isinstance(sel_type, slice):
        return sel_type,sel_type
    elif isinstance(sel_type, str):
        lat_split, lon_split, cnt= list(map(int,sel_type.split("-")))
        lat_size, lon_size = int(384/lat_split), int(768/lon_split)
        lat_cnt, lon_cnt = np.unravel_index(cnt-1, (lat_split,lon_split)) # cnt starts from 1
        return slice(lat_cnt*lat_size,(lat_cnt+1)*lat_size,None), slice(lon_cnt*lon_size,(lon_cnt+1)*lon_size,None)

def get_time(date):
    # Prepare normalized Time input
    date_j = date.to_julian_date()
    time_scales= [1, 365]
    time_sin = [np.sin(date_j*2*np.pi/period)*2.83/2 for period in time_scales] #25,26
    time_cos = [np.cos(date_j*2*np.pi/period)*2.83/2 for period in time_scales] #27,28
    time_h_m = [(date.hour-9)/6.71, (date.month-6.5)/3.45] #29,30
    time_in  = np.array(time_sin+time_cos+time_h_m, dtype=np.float32)
    
    return time_in

def get_latlon(file_f):
    # Prepare normalized latlon
    lons_m, lats_m = np.meshgrid(file_f.grid_xt.values-180, file_f.grid_yt.values/51.96)
                     #(file_f.lon.values-180)/103.92, file_f.lat.values/51.96
    lons_sin = np.sin(lons_m*2*np.pi/360)*2.83/2
    lons_cos = np.cos(lons_m*2*np.pi/360)*2.83/2
    latlon_in = np.array([lons_m,lats_m,lons_sin,lons_cos], dtype=np.float32)
    
    return latlon_in

def dataset_to_tensor_list(file):
    vals = []
    for var in list(file.data_vars):
        var_coords = list(file[var].coords)
        if   ('pfull' in var_coords) and (len(var_coords)>1):
            vals.append(torch.tensor(file[var].values[0]))
        elif (len(var_coords)==0):
            vals.append(torch.tensor(file[var].values[None]))
        elif ('pfull' not in var_coords) or (len(var_coords)==1):
            vals.append(torch.tensor(file[var].values))

    return vals
