# Loading package
import os
import itertools
import sys
import time
from copy import deepcopy
from torchmetrics.functional import mean_squared_error
import logging
logging.basicConfig(level=logging.DEBUG)

import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.multiprocessing import Pool
from torch.utils import data
from torch.utils.data import Subset, DataLoader, SubsetRandomSampler
from test_train_valid_splits import test_train_valid_splits
import random

# Loading self-defined package
from model import CONV2D

torch.set_num_threads(int(os.cpu_count()/2))

# dataset location
dataDir='/scratch2/BMC/gsienkf/Sergey.Frolov/fromStefan/'
ddd=dataDir+'npys_sergey2/ifs'                            

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
            vars_out, testset, kernel_sizes, 
            channels, n_conv, p, bs, loss_name, lr, wd, 
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
    train_valid_set = Dataset(idx_include=train_valid_slice,vars_out=vars_out,)
    
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
    train_inds.sort()
    valid_inds.sort()
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
    # if rank is specified using torch.device, then use multiple GPU for training
    # if rank is an intger, then send to a specific GPU. 
    if type(rank)==torch.device:
      logging.info('distributing training on device: {}'.format(rank))
      model= nn.DataParallel(model)
    model.to(rank) # send model to gpu

    logging.info('rank: {}, setting up loss'.format(rank))
    # define loss functions
    if loss_name == 'mse':
        criterion = torch.nn.MSELoss(reduction='mean')
    elif loss_name == 'mae':
        criterion = torch.nn.L1Loss(reduction='mean')
    elif loss_name == 'wnew':
        # Assume that the norm^2 for the first term in the loss function is about 1 and the norm^2 for the second term is about 1e8
        # then w_weight/1e8 allows us to specify w_weight in a more inuitive units
        w_weight = wd*1e8
        wd = 1e-08
        criterion = torch.nn.MSELoss(reduction='mean')
    else:
        logging.error('rank: {}, Loss not supported!!'.format(rank))
        exit()

    logging.info('rank: {}, setting up optimizer'.format(rank))
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=wd, amsgrad=False) # initialize optimizer
    
    if os.path.isfile(checkfile):
        # read in previous checkpoint file (if exists) to continue training after being interrupted.
        if type(rank)==torch.device:
          map_location = rank
        else:
          map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(checkfile,map_location=map_location)
            
        model.load_state_dict(checkpoint['model_state_dict'])
        if len(checkpoint['optimizer_state_dict']) : 
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

    # initialize W0
    model_0 = deepcopy(model)
    model_0.to(rank)

    max_epoches = 150 # maximum training epoches before forced termination
    patience = 10     # maximum number of consecutive epoches that does not decrease the valid loss (for early stopping)
    
    
    ######################################################################    
    # TRAINING
    try:# prevent raising OOM, which leaves no checkpoint file.

        logging.info('rank: {}, Start training'.format(rank))
        np.random.seed(1337) # ensure reproducibility. for the random sampling

        for epoch in range(epo_init, max_epoches): # loop through training epoches

            ###########
            ## Training
            train_loss = 0
            train_wpen = 0
            train_loss_all = 0
            start_time = time.time()
            model.train() # set model in training mode

            for batch_id, (X, y) in enumerate(train_loader): # loop through mini batches of the training dataset
                batch_time = time.time()
                X, y = X.to(rank), y.to(rank) # send data to gpu
                optimizer.zero_grad()         # reset gradient
                y_pred = model(X)             # get output from model

                loss = criterion(y_pred, y)
                train_loss += loss.item()     # sum loss for all batches

                if loss_name=='wnew':
                  w_penalty = w_weight*sum([mean_squared_error(p.weight, model_0.module.convs[i].weight)  
                                            for i, p in enumerate(model.module.convs)])
                  loss = loss + w_penalty
                else: 
                  w_penalty = 0
                train_wpen += w_penalty

                train_loss_all += loss.item()

                loss.backward()               # compute gradient
                optimizer.step()              # update parameter

            train_loss /= len(train_inds) # compute mean loss
            train_wpen /= len(train_inds) # compute mean loss
            train_loss_all /= len(train_inds) # compute mean loss
            train_losses.append(train_loss_all)           # record training loss

            logging.info("rank: {}, Train epoch: {}, Loss All: {:.4f}, mse loss: {}, wpen: {}, Current Minimum: {:.4f} at epoch# {}".format
                              (rank, epoch, train_loss_all, train_loss, train_wpen, min(train_losses), np.argmin(train_losses)))

            #############
            ## Validation
            with torch.set_grad_enabled(False):   # disable gradient
                model.eval()                      # set model in evaluation mode
                valid_loss = 0
                for X, y in valid_loader:         # loop over every sample in validation set
                    X, y = X.to(rank), y.to(rank) # send data to gpu
                    y_pred = model(X)             # evaluate model
                    loss = criterion(y_pred, y)
                    if loss_name=='wnew':
                      w_penalty = w_weight*sum([mean_squared_error(p.weight, model_0.module.convs[i].weight)
                                                for i, p in enumerate(model.module.convs)])
                      loss = loss + w_penalty
                    valid_loss += loss.item()
#                    y_diff_s = (y_pred - y)**2    # compute loss
#                    valid_loss += torch.sum(y_diff_s).item() # sum all loss
                    
            valid_loss /= len(valid_inds)    # average loss
            valid_losses.append(valid_loss)  # record validation loss

            logging.info("rank: {}, Valid epoch: {}, Loss: {:.4f}, Current Minimum: {:.4f} at epoch# {}".format(rank, epoch, valid_loss, min(valid_losses), np.argmin(valid_losses)))

            impatience = epoch - np.argmin(valid_losses) # check number of epoches from last validation loss minimum
            if min(valid_losses) == valid_loss: # if current epoch has the minimal valid loss
                best_model = deepcopy(model.state_dict()) # extract model parameter
                best_optim = deepcopy(optimizer.state_dict()) # extract optimizer state

                # save history and current state to checkpoint file
                # for paralel GPU, this step can dominate the cost for small data batches
                if type(rank)!=torch.device:
                  logging.info('rank: {}, Reach new min, Saving checkpoint \n'.format(rank))
                  _checkpoint_(rank,checkfile,best_model,best_optim,epoch,train_losses,valid_losses,impatience,max_time)

            check_gpu(rank) # check gpu status
            current_time = time.time()
            elapsed_time = current_time - start_time
            logging.info("rank: {}, Elapsed time: {:.1f} \n".format(rank,elapsed_time))

            max_time = int(max(max_time, elapsed_time)) # update maximum time for each epoch

            # break from training if there have been too many consecutive epoches that does not decrease the valid loss
            if impatience > patience:
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
    def __init__(self, idx_include=slice(0,None), # skip first 40 samples
                       vars_out='t',
                       **kwargs):
        # slicing output variables
        if vars_out == 't':
            slice_out = slice(0,127)
        elif vars_out == 'u':
            slice_out = slice(127*1,127*2)
        elif vars_out == 'v':
            slice_out = slice(127*2,127*3)
        elif vars_out == 'q':
            slice_out = slice(127*3,127*4)
        elif vars_out == 'ps':
            slice_out = slice(127*4,127*4+1)
        
        t = time.time()
            
        self.ins = []
        # load data in np array and cast it as a torch object
        # 4D dataset [batch_size, channels, height, width]
        #breakpoint()
        f06_in = np.load(ddd+'_f06_ranl_sub')[idx_include]
        self.ins = torch.from_numpy(np.copy(f06_in))
        self.ndates, _, self.nlat, self.nlon = f06_in.shape # get data shape
        del(f06_in)

        out    = np.load(ddd+'_out_ranl_sub')[idx_include,slice_out]
        self.out = torch.from_numpy(np.copy(out))
        del(out)
        
        print('Channel in  size: {}'.format(self.ins.shape[1]))
        print('Channel out size: {}'.format(self.out.shape[1]))
        print('Time snapshots: {}'.format(self.ndates))
        print('time preparing data: {}s'.format(time.time()-t))
        
    def __normal_in__(self,x):
        '''normalize input data'''
        return x
    def __normal_out__(self,x):
        '''normalize output data'''
        return x
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
def reset_network(old_name,new_name,rank=''):
  # read old checkpoint, reset training hsitory to scratch
  print(f"reading {old_name}")
  model = torch.load(old_name)
  print(f"reseting to {new_name}")
  _checkpoint_(rank=0, checkfile=new_name, best_model=model['model_state_dict'], best_optim=[], epoch=0,
              train_losses=[], valid_losses=[], impatience=0, max_time=0)
  del model

def create_checkpoint_filename(params,dirname='checks',prefix='conv2d'):
  # filename for training checkpoints
  fn = dirname+'/'+prefix+'_'+'_'.join([str(elem) for elem in params]) # filename for training checkpoints
  return fn

def compute_skill(y, y_pred):
   y_pred_ts=y_pred.cpu().detach().numpy().view().reshape((y.shape[0], np.prod(y.shape[1:])))
   y_ts=y.detach().numpy().view().reshape((y.shape[0], np.prod(y.shape[1:])))
   skill = 1-np.mean((y_pred_ts-y_ts)**2,axis=1)/np.mean((y_ts)**2,axis=1)
   return skill

