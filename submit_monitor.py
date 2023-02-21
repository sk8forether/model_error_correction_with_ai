import os
import sys

userid     = os.environ.get("USER") 
#main_dir = "/home/Sergey.Frolov/work/model_error/code/model_error_correction"
#python_exe = "/scratch1/NCEPDEV/global/Tse-chun.Chen/anaconda3/envs/ltn/bin/python"
main_dir = "./"
if not os.path.exists(main_dir+'/jobs/'):
  os.mkdir(main_dir+'/jobs/')
if not os.path.exists(main_dir+'/slurm_out/'):
  os.mkdir(main_dir+'/slurm_out/')
python_exe = os.environ.get('MYPYTHON')
slurm_account = os.environ.get('SLURM_ACCOUNT')

def monitor():
    '''Submit and Monitor individual training tasks. 
    This is needed for clusters' scheduler environment:
    * limited GPU access time. training time may exceed the limit.
    * limited number of submitted jobs. ''' 
    
    import subprocess
    import itertools
    import logging
    logging.basicConfig(level=logging.DEBUG)
    from time import time
    from joblib import Parallel, delayed
    
    from torch import load
    import numpy as np
    import json


    # define hyperparameter search space
    with open('params_space.dict') as json_file:
      params_space=json.loads(json_file.read())

    # construct param_list, which contains all combinations of hyper search space.
    keys = params_space.keys()
    spaces = [params_space.get(i) for i in keys]
    space_list_full = list(itertools.product(*spaces))
    
    space_list_full.sort(key=lambda x:(x[6],x[7],[5])) # sort by chennels, n_conv, kernel_sizes

    logging.info("There are {} of combinations!!".format(len(space_list_full)))

    try: # avoid error when no current running job
        
        # identify still running jobs to avoid assigning same jobid
        files = subprocess.run(f'squeue | grep {userid}| grep job ',shell=True,capture_output=True,check=True).stdout.decode().split()[2::8]
        running_ids = [int(f[3:6]) for f in files]
        running_files = [main_dir+'/jobs/'+f[:6]+'.py' for f in files]
        
    except subprocess.CalledProcessError:
        logging.info("No jobs running currently!")
        files = []
        running_ids = []
        running_files = []
        
    avail_ids = list(range(1000)) # create a large pool of job id. 1000 is just a big number
    for j in running_ids:
        avail_ids.remove(j) # remove currently using job id from the pool
        
    # grap param_list from the running_files
    running_params=[]
    for f in running_files:
        with open(f,'r') as reader:
            for i,line in enumerate(reader):
                if i ==3:
                    # for variable scoping
                    lcls=locals()
                    exec(line,  globals(), lcls)
                    param_list = lcls["param_list"]

            running_params += param_list

    logging.info('running ids: {}'.format(running_ids))
    if len(running_params) > 0: 
        logging.info('still searching: {}'.format([list(np.unique(np.array(running_params).T[i])) for i in range(len(running_params[0]))]))
        
    logging.info('total running ids: {}'.format(len(running_ids)))
    logging.info('skipping still running combinations: {}'.format(len(running_params)))

    # filter out the ones done with training
    def fn(l):
        '''Return hyper combinations that needs to be (re)submitted.'''
        checkfile = './checks/conv2d_'+'_'.join([str(elem) for elem in l])
        if not (l in running_params): # skip the still running jobs
            if os.path.isfile(checkfile):
                try: # catch error from still writting file
                    file = load(checkfile, map_location='cpu')
                    impatience = file['impatience']
                    epoch = file['epoch']
                    if (impatience < 20) & (epoch < 499):
                        return l
                except (RuntimeError, EOFError):
                    logging.info("Failed reading: " + checkfile)
                    return l
            else:
                return l

    t0 = time()
    space_list = Parallel(n_jobs=20,
                          verbose=10,)(delayed(fn)(l) for l in space_list_full) # get list of hyper combinations to submit using multi-thread parallel jobs
    space_list = [i for i in space_list if i] # get rid of None from no-return

    logging.info('Took {}s to scan through all combinations'.format(time()-t0))
    logging.info("{}/{} training to be (re)submitted".format(len(space_list),len(space_list_full)))
    
    # avoid having 400+jobs in queue
    jobs_queued   = len(running_ids)             # get number of jobs already in queue
    jobs_tosubmit = len(space_list)/8            # get number of jobs to be submitted (each job/node runs 8 tasks/gpus)
    if jobs_queued + jobs_tosubmit > 400:
        jobs_allowed = 400 - jobs_queued         # get number of jobs that are really submitting
        space_list = space_list[:jobs_allowed*8]
        
        logging.info("Jobs in queue {} plus jobs to be submitted {} exceeds 400. Cutting off 400+ jobs".format(jobs_queued,jobs_tosubmit))

    
    list_batches = [space_list[i:i+8] for i in range(0,len(space_list),8)] # divide param_list to batches of 8 (number of batches = number of submitting jobs)
    
    logging.info("There are {} of batches!!".format(len(space_list)/8))

    # submit jobs
    for n,b in enumerate(list_batches):
        
        # prepare job file for submission
        lines = ["import sys \n",
                 f"sys.path.append('{main_dir}') \n",
                 "from training import Train_CONV2D \n",
                 "param_list = {} \n".format(b),
                 "Train_CONV2D(param_list) \n"]                          
        jobid   = avail_ids[n]
        
        logging.info('submitting to jobid: {}'.format(jobid))
        
        jobdir  = './jobs/'
        jobfile = 'job{:03d}.py'.format(jobid)
        with open(jobdir+jobfile,'w') as fh:
            fh.writelines(lines)

        os.system(f"> {main_dir}/slurm_out/{jobfile}.out") # clear previous log

        submitline = "sbatch -t 30:0:0 -A {} -p fgewf --qos=windfall -N 1 --job-name {} --output {}/slurm_out/{}.out --wrap '{} -u {}' ".format(slurm_account, jobfile, main_dir, jobfile, python_exe, jobdir+jobfile) # job submit line. modify as needed.
        os.system(submitline) # submit job
        
if __name__ == "__main__":
    monitor()
