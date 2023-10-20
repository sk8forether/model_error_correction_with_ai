import numpy as np

dataDir='/scratch2/BMC/gsienkf/Sergey.Frolov/fromStefan/'   # directory for input data
npyDir=dataDir+'npys_sergey2/ifs_'                          # output directory

#for i in ['f06_ranl_sub','out_ranl_sub']:
for i in ['out_ranl_sub']:
    file = npyDir+i
    a = np.load(file, mmap_mode='r')
    
    mean = a.mean(axis=(0,2,3))
    std  = a.std(axis=(0,2,3))
        
    np.save(npyDir+i+'_mean_1d', mean)
    np.save(npyDir+i+'_std_1d', std)
