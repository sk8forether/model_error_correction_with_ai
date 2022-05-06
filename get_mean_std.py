import numpy as np

ddd = './npys/ifs_'
nbc=21
for i in ['f06_ranl_low','sfc_ranl_low','out_ranl_low','f06_ranl_sub','sfc_ranl_sub','out_ranl_sub']:
    file = ddd+i
    a = np.load(file, mmap_mode='r')[:1460]
    
    mean = a.mean(axis=(0,2,3))
    std  = a.std(axis=(0,2,3))
        
    np.save(ddd+i+'_mean_1d', mean)
    np.save(ddd+i+'_std_1d', std)
