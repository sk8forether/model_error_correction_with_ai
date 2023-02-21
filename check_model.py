import os
import sys
import logging
from time import time
from glob import glob
from joblib import Parallel, delayed
#logging.basicConfig(level=logging.DEBUG)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from netCDF4 import Dataset
from test_train_valid_splits import test_train_valid_splits

#main_dir = "/scratch2/BMC/gsienkf/Tse-chun.Chen/for_sergey/model_error_correction"
#main_dir = "/home/Sergey.Frolov/work/model_error/code/model_error_correction/"
#python_exe = "/scratch1/NCEPDEV/global/Tse-chun.Chen/anaconda3/envs/ltn/bin/python"
main_dir = "./"
python_exe = os.environ.get('MYPYTHON')
slurm_account = os.environ.get('SLURM_ACCOUNT')

def int_float_str(s):
    '''
    convert string to int or float if possible
    '''
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s

def count_parameters(model):
    '''count number of parameters in the NN model'''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_train_param_name(model_type):
    logging.info('################################################')
    logging.info('## Get_train_param_name                         ')
    logging.info('################################################')
    
    if model_type == 'conv2d':

        from training import _train_ as Train
        arg_count = Train.__code__.co_argcount-1 # count training code arguments
        arg_names = Train.__code__.co_varnames[1:arg_count+1] # get training code variable names
        
    else:
        logging.error(model_type+" not supported!")
    return arg_count, arg_names

def get_test_dataset(hyperparam, num_workers=0):
    from training import Dataset_np as Dataset

    logging.info('################################################')
    logging.info('## get_test_dataset                             ')
    logging.info('################################################')
    
    # define the test index range
    testset = hyperparam['testset']
    splits = test_train_valid_splits(testset)
    test_slice = splits["test_slice"]
        
    test_set = Dataset(idx_include=test_slice, **hyperparam) # initiate dataset object
    test_Loader = DataLoader(test_set, batch_size=len(test_set),num_workers=num_workers) # set up data loader
    return test_Loader

def get_train_dataset(hyperparam, num_workers=0):
    from training import Dataset_np as Dataset

    logging.info('################################################')
    logging.info('## get_test_dataset                             ')
    logging.info('################################################')

    # define the training and validation index range
    testset = hyperparam['testset']
    splits = test_train_valid_splits(testset)
    train_valid_slice = splits["train_valid_slice"]

    train_valid_set = Dataset(idx_include=train_valid_slice, **hyperparam) # initiate dataset object
    train_valid_Loader = DataLoader(train_valid_set, batch_size=hyperparam['bs'], num_workers=num_workers) # set up data loader
    return train_valid_Loader


def get_norm(filename):
    logging.info('################################################')
    logging.info('## get_norm                                     ')
    logging.info('################################################')

    hyperparam = read_hyperparam(filename) # get hyperparameter from filename
    test_Loader = get_test_dataset(hyperparam, 4) 
    return test_Loader.dataset.mean_out.numpy(), test_Loader.dataset.std_out.numpy()


def read_hyperparam(filename):
    logging.info('################################################')
    logging.info('## read_hyperparam                              ')
    logging.info('################################################')
 
    logging.info(filename) # full list
    
    name = filename.split('/')[-1]                         # get rid of parent directory
    hyper = [int_float_str(i) for i in name.split('_') ]   # get hyperparameters
    model_type = hyper[0]
    
    arg_count, arg_names = get_train_param_name(model_type)
    
    if len(hyper) != arg_count+1: # check if the argument counts and filename matches
        sys.exit("format incorrect!!: {}".format(name))
    
    hyperparam = dict(zip(('model_type',)+arg_names, hyper))
    return hyperparam

def read_checkfile(filename):
    '''read checkpoint file for hyperparameters and training status'''
    logging.info('################################################')
    logging.info('## read_checkfile                               ')
    logging.info('################################################')
    
    hyperparam = read_hyperparam(filename) # get hyperparameter from filename
    
    try:
        checkfile = torch.load(filename,map_location=torch.device('cpu')) # load checkpoints from file on cpu
        valid_min = min(checkfile['valid_loss'])
        epoches = len(checkfile['valid_loss'])
        impatience = checkfile['impatience']
        add_param = dict(zip(['filename','epoches','impatience','valid_min'],
                             [filename, epoches, impatience, valid_min]))
        hyperparam.update(add_param)
        return hyperparam
    
    except (RuntimeError, EOFError):
        logging.error("Failed reading: " + filename)
    except (ValueError):
        logging.error("OOM: " + filename)
       
def model_to_nc(filename, if_return_nc=False, if_norm=True,):
    logging.info('################################################')
    logging.info('## output coeff. to nc file                     ')
    logging.info('################################################')

    checkfile = torch.load(filename, map_location=torch.device('cpu'))

    coeff = list(checkfile['model_state_dict'].items())

    # get a list of sizes of each layer
    n_nodes = [coeff[0][1].numpy().shape[1]]
    for n in range(1,len(coeff),2):
        n_nodes.append(coeff[n][1].numpy().shape[0])

    logging.info(n_nodes)

    ncfile = Dataset('nn.nc', mode='w', format='NETCDF4')
    ncfile.title    = 'NN coefficients'
    ncfile.subtitle = filename
    ncfile.nn_sizes  = np.array(n_nodes)
    hyperparam = read_hyperparam(filename)
    ncfile.var_out  = hyperparam['vars_out']

    if if_norm:
        loader   = get_test_dataset(hyperparam)
        mean_in  = loader.dataset.mean_in.numpy().squeeze()
        std_in   = loader.dataset.std_in.numpy().squeeze()
        mean_out = loader.dataset.mean_out.numpy().squeeze()
        std_out  = loader.dataset.std_out.numpy().squeeze()

    # create nc dimension
    for n,s in enumerate(n_nodes):
        ncfile.createDimension(f'layer{n}', s)

    # create nc variables
    precision = np.float32
    contiguous = True
    for n in range(int(len(coeff)/2)):
        w = coeff[n*2][1].numpy().squeeze()
        b = coeff[n*2+1][1].numpy().squeeze()

        if (n==0) & if_norm: # input normalization
            wt = ncfile.createVariable(f'w{n}', precision, (f'layer{n}',f'layer{n+1}'),contiguous=contiguous)
            wt[:,:] = precision((w/std_in).T)
            bs = ncfile.createVariable(f'b{n}', precision, (f'layer{n+1}'),contiguous=contiguous)
            bs[:] = precision(-np.dot(w,mean_in/std_in) + b)
        elif (n==int(len(coeff)/2)-1) & if_norm: # output normalization
            wt = ncfile.createVariable(f'w{n}', precision, (f'layer{n}',f'layer{n+1}'),contiguous=contiguous)
            wt[:,:] = precision(w.T*std_out)
            bs = ncfile.createVariable(f'b{n}', precision, (f'layer{n+1}'),contiguous=contiguous)
            bs[:] = precision(b*std_out + mean_out)
        else:
            wt = ncfile.createVariable(f'w{n}', precision, (f'layer{n}',f'layer{n+1}'),contiguous=contiguous)
            wt[:,:] = precision(w.T)
            bs = ncfile.createVariable(f'b{n}', precision, (f'layer{n+1}'),contiguous=contiguous)
            bs[:] = precision(b)

    logging.info(ncfile)

    if if_return_nc:
        return ncfile
    else:
        ncfile.close()

 
def read_model(filename, if_hyperparam=False, if_iosize=False):
    '''initialize model from checkpoint file'''
    logging.info('################################################')
    logging.info('## read_model                                   ')
    logging.info('################################################')
    
    hyperparam = read_hyperparam(filename) # get hyperparameter from filename
    
    checkfile = torch.load(filename, map_location=torch.device('cpu')) # load checkpoints from file on cpu
    
    input_size  = checkfile['model_state_dict']['convs.0.weight'].shape[1]
    output_size = checkfile['model_state_dict']['convs.{}.weight'.format(hyperparam['n_conv']-1)].shape[0]
    
    if hyperparam['model_type'] == 'conv2d':
        from model import CONV2D as NN
    
    model = NN(input_size=input_size, output_size=output_size, **hyperparam) # initialize model object
    model.load_state_dict(checkfile['model_state_dict']) # load state from checkpoint file into the model object

    if if_hyperparam and if_iosize:
        return model, hyperparam, (input_size,output_size)
    elif if_hyperparam and not if_iosize:
        return model, hyperparam
    elif not if_hyperparam and if_iosize:
        return model, (input_size,output_size)
    else:
        return model

def eval_model(filename):
    '''evaluate model skill on testing data'''
    logging.info('################################################')
    logging.info('## eval_model                                   ')
    logging.info('################################################')

    model, hyperparam = read_model(filename, True) # get model
    test_Loader = get_test_dataset(hyperparam, 4) # get testing data loader
    
    # name the truth and prediction files
    name = filename.split('/')[-1] 
    y_pred_file = main_dir+'/npys/ypred_'+name+'.npy' # prediction from model
    y_file = main_dir+'/npys/y_'+name+'.npy'          # truth
    logging.info(y_pred_file)
    logging.info(y_file)
    
    t0 = time()
    # run model through the testing dataset in evaluation mode
    with torch.set_grad_enabled(False):
        model.eval()
        for X, y in test_Loader:
            y_pred = model(X)
    logging.info('took {}s'.format(time()-t0))

    np.save(y_pred_file, y_pred)
    np.save(y_file, y)
    
def sub_eval_model(filename, if_get_norm=False, if_renew=True, if_wait=True):
    '''Submit eval_model job to slurm'''
    
    name = filename.split('/')[-1] 
    y_pred_file = main_dir+'/npys/ypred_'+name+'.npy'
    y_file = main_dir+'/npys/y_'+name+'.npy'
    if_file_exist = (os.path.isfile(y_pred_file) & os.path.isfile(y_file))
    if_submit = not if_file_exist or if_renew # determine if read from previous results or submit job for renewal or for first time
    
    if if_submit:
        logging.info("filename")
        logging.info('model eval 1st time. will take longer.')
        
        # make tmp.py file for job submission 
        os.system("echo from check_model import eval_model > ./tmp.py")
        os.system('''echo "eval_model(\'{}\')" >> {}/tmp.py'''.format(filename,main_dir))

        # put together submit command
        prefix = 'sbatch'
        if if_wait: # block the current process to wait for the job to complete
            prefix = 'sbatch --wait'
        submitline = prefix + f' -t 30:0 -A {slurm_account} -p bigmem -N 1 --output {main_dir}/eval_model.out --wrap "{python_exe} -u  {main_dir}/tmp.py " '
        os.system(submitline)
    
    else:
        logging.info('model evaled before. reading from file.')
        
    if_output = if_wait or not if_submit # output the truth and prediction if waiting for the job to complete or just to read previous results
    if if_output:
        y_pred = np.load(y_pred_file)
        y = np.load(y_file)

        logging.info('finished')
        logging.info('Learned percentage: {}'.format(1-np.mean((y_pred-y)**2)/np.mean((y)**2)))
        logging.info(f'R2:  {1-np.mean((y_pred-y)**2)/np.mean((y-y.mean())**2)}')
        logging.info(f'MSE: {np.mean((y_pred-y)**2)}')
    
    if if_get_norm & if_output:
        mean_out, std_out = get_norm(filename)
        return y_pred, y, mean_out, std_out
    
    elif if_output:
        return y_pred, y
    else:
        logging.info('not waiting for sbatch. keep going')
    
    
def collect_models():
    ''' collect the hyperparameters and training status for all the trained models into dataframe'''
    logging.info('################################################')
    logging.info('## collect_models                               ')
    logging.info('################################################')
    
    t0 = time()
    checks = sorted(glob(f'{main_dir}/checks/conv*'))

    dicts = Parallel(n_jobs=10, verbose=10)(delayed(read_checkfile)(filename) for filename in checks) # read the checkpoint file in parallel
    dicts = [i for i in dicts if i] # get rid of None

    df = pd.DataFrame(dicts) # convert to dataframe
    df.keys()

    df.to_pickle(f'{main_dir}/checks/df_low-res-config') # save the dataframe to file
    logging.info('took {} to finish building dataframe'.format(time()-t0))

def sub_collect_models(if_renew=True, df='df_low-res-config', if_wait=True):
    ''' Submit collect_models to slurm'''
    
    if if_renew:
        prefix = '''sbatch'''
        if if_wait:
            prefix = '''sbatch --wait'''
        submitline = prefix+f''' -t 120:0 -A {slurm_account} -p hera -N 1 --output {main_dir}/collect_models.out --wrap "{python_exe} -u -c 'from check_model import collect_models
collect_models()'  " '''
        os.system(submitline)

    else:
        logging.info('reading df from file.')
    df = pd.read_pickle('{}/checks/{}'.format(main_dir,df))
    return df


def saliency(filename,):
    ''' Compute averaged gradients in training dataset and Save to file'''
    name = filename.split('/')[-1] # get saliency filename from checkpoint filename
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # use gpu if available. use cpu if not (significantly slower)
    model, hyperparam, iosize = read_model(filename, if_hyperparam=True, if_iosize=True) # get trained model
    model.to(device) # put model to device (gpu or cpu)
    model.eval() # set model in evaluation mode
    
    hyperparam['bs']=1 # to avoid unmatching batchsize at the end

    train_valid_Loader = get_train_dataset(hyperparam, num_workers=8) # set up training data loader
    
    J_tmp = np.zeros(iosize[::-1])
    J = np.zeros(iosize[::-1])

    for j, (x,y) in enumerate(train_valid_Loader):
        logging.info(j)
        x = x.to(device)
        x.requires_grad=True
        y = model(x)

        # pytorch was designed to scalar output
        # so the backward doesn't provide full Jacobian matrix, but a product of vector^T and J
        # need to loop through the output dimension to get the full matrix
        for i in range(127):
            ext = torch.zeros((hyperparam['bs'],iosize[1],32,64),device=device)
            ext[:,i] = 1.
            y.backward(gradient=ext, retain_graph=True)
            J_tmp[i,:] = x.grad.data.cpu().mean(axis=(0,2,3))
            _ = x.grad.data.zero_()

        J += J_tmp

    J /= j
    np.save('{}/npys/J_{}'.format(main_dir,name),J)

        
def sub_saliency(filename, if_renew=False):
    '''Submit saliency to slurm'''
    
    name = filename.split('/')[-1] 
    J_file = '{}/npys/J_{}.npy'.format(main_dir,name)

    
    if not os.path.isfile(J_file) or if_renew:
        logging.info('saliency eval 1st time. will take longer.')
        os.system(f"echo from check_model import saliency > {main_dir}/tmp.py")
        os.system('''echo "saliency(\'{}\')" >> {}/tmp.py'''.format(filename,main_dir))

        submitline = f'sbatch --wait -t 30:0:0 -p fgewf --qos=windfall -N 1 --output {main_dir}/eval_saliency.out --wrap "{python_exe} -u  {main_dir}/tmp.py " '
        os.system(submitline)
    
    else:
        logging.info('saliency evaled before. reading from file.')
        
    J = np.load(J_file)
    logging.info('finished')
    
    return J
    

    
