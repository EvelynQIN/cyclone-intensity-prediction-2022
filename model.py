import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from tcn import TemporalConvNet

class ConvolutionPoolLayer(torch.nn.Module):
    """
        Convolutional layer with optional MaxPooling and optional activation.
    """
    def __init__(self, in_channels, filter_size, out_channels, layer_name, activation, bias=True, use_pooling=True):
        """
        Constructor for ConvolutionPool layer.
        :param in_channels: Number of channels of the input.
        :param filter_size: Width and height of the square filter (scalar).
        :param out_channels: How many feature maps to produce with this layer.
        :param layer_name: A name for this layer.
        :param activation: Activation function used on the output of the layer.
        :param bias: If set to False, the layer will not learn an additive bias. Default: True.
        :param use_pooling: Use 2x2 max-pooling if True. Default: True.
        """   
        super().__init__()
        # Convolution parameters
        self.stride = (1, 1) 
        self.filter_size = filter_size
        # Convolution operation - we do the padding manually in order to get tensorflow 'same' padding
        self.conv = torch.nn.Conv2d(in_channels, out_channels, self.filter_size, stride=self.stride, padding=0, bias=bias)
        # Use pooling to down-sample the image resolution?
        # This is 2x2 max-pooling, which means that we consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        self.max_pool = torch.nn.MaxPool2d(2, 2) if use_pooling else None
        # This adds some non-linearity to the formula and allows us to learn more complicated functions.
        self.activation = activation


    def get_padding_amount(self, shape):
        """
        Computes the amount of padding so that the input size is equal to output size.
        PyTorch doesn't provide 'same' padding so we use the implementation from TensorFlow.
        """
        _, _, input_h, input_w = shape
        output_h = int(math.ceil(float(input_h) / float(self.stride[0])))
        output_w = int(math.ceil(float(input_w) / float(self.stride[1])))
         
        if input_h % self.stride[0] == 0:
            pad_along_height = max((self.filter_size - self.stride[0]), 0)
        else:
            pad_along_height = max(self.filter_size - (input_h % self.stride[0]), 0)
        if input_w % self.stride[1] == 0:
            pad_along_width = max((self.filter_size - self.stride[1]), 0)
        else:
            pad_along_width = max(self.filter_size - (input_w % self.stride[1]), 0)
            
        pad_top = pad_along_height // 2 # amount of zero padding on the top
        pad_bottom = pad_along_height - pad_top     # amount of zero padding on the bottom
        pad_left = pad_along_width // 2     # amount of zero padding on the left
        pad_right = pad_along_width - pad_left  # amount of zero padding on the right

        return pad_left, pad_right, pad_top, pad_bottom

    def forward(self, x):
        """
        Compute the forward pass of the ConvolutionPoolLayer layer.
        :param x: The input tensor.
        :return: The output of this layer. 
        """
        padding = self.get_padding_amount(x.shape)
        x = torch.nn.functional.pad(x, padding)  # [left, right, top, bot]
        x = self.conv(x)
        if self.max_pool:
            x = self.max_pool(x)
        if self.activation:
            x = self.activation(x)
        return x

class ConvNet(torch.nn.Module):
    """
    Concat the cnn vector of each timestep 
    """
    def __init__(self, feature_map_sizes, filter_sizes, activation=torch.nn.ReLU()):
        """
        Constructor for ConvNet.
        :param feature_map_sizes: list of out_channels for the convolution layers.
        :param filter_sizes: list of filter_size for each convolution layers.
        :param activation: Activation function used in the network. Default: ReLU.
        """   
        super().__init__()
        # Flatten layer
        self.flatten = torch.nn.Flatten() 
        # Fully Connected
        #self.fc = FullyConnectedLayer(1152, 10, 'dense_layer', activation=None)
        # Softmax
        # self.softmax = torch.nn.Softmax(dim=1)
        # Convolutions
        in_channels = 7
        self.convolutions = torch.nn.ModuleList()
        for i, (out_channels, filter_size) in enumerate(zip(feature_map_sizes, filter_sizes)):
            self.convolutions.append(ConvolutionPoolLayer(in_channels, filter_size, out_channels, f'conv{i}_layer', activation))
            in_channels = out_channels
   

    def forward(self, x):
        """
        Compute the forward pass of the network.
        :param x: The input tensor. (batch_size * timesteps * 7channels * 11 * 11)
        :return: The activated output of the network. 
        """

        # concatenate the feature vector of each time step
        x = x.transpose(0, 1) # expected dim: timesteps * batch_size * 7channels * 11 * 11

         # iterate over all timestep
        xt = x[0]
        for conv in self.convolutions:
            xt = conv(xt)
        xt = self.flatten(xt)
        cnn_stack = xt.reshape(xt.shape[0], xt.shape[1], 1)  # batch_size * vector_length * 1
        for t in range(1, x.shape[0]):
            xt = x[t]
            for conv in self.convolutions:
                xt = conv(xt)
            xt = self.flatten(xt)  # batch_size * vector_length
            xt = xt.reshape(xt.shape[0], xt.shape[1], 1) # batch_size * vector_length * 1 n
            cnn_stack = torch.cat((cnn_stack, xt), dim = 2) # batch_size * vector_length * timesteps 
        
        #print("shape of x: {}".format(cnn_stack.shape)) test dim 
            
        return cnn_stack

class TCN(torch.nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = torch.nn.Linear(num_channels[-1], output_size)
        self.init_weights()
        self.feature_map_sizes = [8, 16]
        self.filter_sizes = [2, 2]
        self.ConvNet = ConvNet(self.feature_map_sizes, self.filter_sizes, activation=torch.nn.ReLU())    

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        meta, ra = x
        cnn_concat = self.ConvNet(ra) # batch_size * vector_length * timesteps    
        meta = meta.transpose(1, 2)
        feature_all = torch.cat((cnn_concat, meta), dim = 1)     
        output = self.tcn(feature_all)
        y = self.linear(output[:, :, -1]) # expected dim:  128 * 6
        # y = torch.tensor([])
        # for t in range(output.shape[2]):
        #     yt = self.linear(output[:, :, t])  # batch_size * 1   # ???????????check 是否需要搞不同的linear layer
        #     y = torch.cat((y, yt), 1)
        #print("shape of y: {}".format(y.shape)) #test dim  128 * 6?
        return y

