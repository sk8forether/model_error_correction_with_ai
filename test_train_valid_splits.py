import json 

def test_train_valid_splits(strategy):

    with open('params_space.dict') as json_file:
      params_space=json.loads(json_file.read())


    if strategy==0:
        test_slice = slice(40+1460,None)
        train_valid_slice = slice(40,40+1460)
    elif strategy==1:
        test_slice = slice(40,40+367)
        train_valid_slice = slice(40+368,None)
    elif strategy==2:
        test_slice = slice(None,None) #for sample use
        train_valid_slice = slice(None,None) #for sample use
    elif strategy==3: # for Sergey; train with the penultimate week. reserve and split last week for 3-day validation and 4-day indp test.
        test_slice = slice(-4*4,None) # for Sergey; indp test with the last 4 days
        train_valid_slice = slice(-14*4,-4*4) # last 14 days to 4th to last day.
    else:
        logging.error("rank: {}, testset strategy values {} not supported".format(rank, strategy))
        exit()

    rv = dict()
    rv["test_slice"] = test_slice
    rv["train_valid_slice"] = train_valid_slice
    return rv
