# Loading package
from torch import cat
import torch.nn as nn
import torch.nn.functional as F

import logging
logging.basicConfig(level=logging.DEBUG)

class CONV2D(nn.Module):
    '''Define NN model class'''
    def __init__(self, 
                 input_size,          # int: input number of channels/neurons
                 output_size,         # int: output number of channels/neurons
                 kernel_sizes="3-2",  # int or str: kernel size. a list of kernel sizes for different layers can be specified in str using "-" separator. 
                 channels="1024",     # int or str: channel number for hidden layers. a list of channel number for different layers can be specified in str using "-" separator.
                 n_conv=2,            # int: number of nn layers
                 p=0,                 # float: dropout probability
                 **kwargs):
        
        super(CONV2D, self).__init__()
        
        if type(kernel_sizes) == str:
            kernel_sizes = [int(i) for i in kernel_sizes.split("-")] # get a list of kernel sizes for each layer
        else:
            kernel_sizes = [kernel_sizes]
            
        if type(channels)     == str:
            channels     = [int(i) for i in channels.split("-")] # get a list of number of channels for each layer
        else:
            channels     = [channels]
        
        self.n_conv = n_conv
        
        if len(channels) != n_conv-1:
            channels *= n_conv-1 # match length of channel list with the number of nn layers, assuming it is the same for all layers
            
        if len(kernel_sizes) != n_conv:
            kernel_sizes *= n_conv # match length of kernel size list with the number of nn layers, assuming it is the same for all layers
        
        in_channels = [input_size] + channels
        out_channels = channels + [output_size]
        
        logging.info('Model setup: {} {} {}'.format(input_size, channels, output_size))
        logging.info('kernels: {}'.format(kernel_sizes))
        
        pad_size = [int((i-1)/2) for i in kernel_sizes] # define padding sizes for each layer according to kernel size, for convolutional nn.
        
        self.convs = nn.ModuleList() # create container for cnn odjects
        self.pads  = nn.ModuleList() # create container for padding odjects
        for k in range(n_conv):
            self.pads.append( nn.ReflectionPad2d((pad_size[k],pad_size[k],0,0)) ) # add reflection padding layers for north-south boundary to container
            self.convs.append( nn.Conv2d(in_channels[k],out_channels[k],
                                         kernel_size=kernel_sizes[k],stride=1,
                                         padding_mode='circular',
                                         padding=(pad_size[k],0) ))               # add cnn layers with circuler padding layers for east-west boundary to container
            
        self.dropout = nn.Dropout(p) # define dropbout layer
        
        self.act = F.relu # define activation function
        
        logging.info("Number of parameters: {}".format(sum(p.numel() for p in self.parameters() if p.requires_grad)))
        
    def forward(self,x):
        '''put the ingredients together'''
        
        for k in range(self.n_conv-1):
            x = self.pads[k](x)
            x = self.dropout(self.act(self.convs[k](x)))
        
        x = self.pads[-1](x)
        x = self.convs[-1](x)
        return x
