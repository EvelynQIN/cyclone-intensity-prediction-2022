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

class TCN_LSTM(torch.nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, lstm_hidden_size, n_layers, device):
        super(TCN_LSTM, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.lstm_hidden_size = lstm_hidden_size
        self.linear = torch.nn.Linear(self.lstm_hidden_size, output_size)
        self.init_weights()
        self.feature_map_sizes = [8, 16]
        self.filter_sizes = [2, 2]
        
        self.n_layers = n_layers
        self.lstm = torch.nn.LSTM(num_channels[-1], self.lstm_hidden_size, num_layers = self.n_layers, batch_first=True)  
        self.ConvNet = ConvNet(self.feature_map_sizes, self.filter_sizes, activation=torch.nn.ReLU())  
        self.device = device
        
        

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x, h):
        meta, ra = x
        cnn_concat = self.ConvNet(ra) # batch_size * vector_length * timesteps    
        meta = meta.transpose(1, 2)
        feature_all = torch.cat((cnn_concat, meta), dim = 1)
        tcn_output = self.tcn(feature_all)
        # print(tcn_output.shape)
        tcn_output = tcn_output.transpose(1,2)
        # print(tcn_output.shape)
        _, (h, _) = self.lstm(tcn_output, h)
        y = self.linear(h[-1]) # expected dim:  128 * 6
        return y

    def init_hidden(self, batch_size):
        hidden = (torch.zeros([self.n_layers, batch_size, self.lstm_hidden_size]).to(self.device), torch.zeros([self.n_layers, batch_size, self.lstm_hidden_size]).to(self.device))
        return hidden

class Lstm(torch.nn.Module):
    """
    para
    - input_size: feature size
    - hidden_size: number of hidden units
    - output_size: number of output
    - num_layers: layers of LSTM to stack
    """
    def __init__(self, input_size_ra, input_size_meta, hidden_size, output_size, num_layers, device, use_ra = True):
        super().__init__()
        self.input_size_ra = input_size_ra
        self.input_size_meta = input_size_meta
        self.output_size = output_size
        self.n_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm_meta = torch.nn.LSTM(input_size_meta, hidden_size, num_layers, batch_first=True).float()
        self.lstm_meta_ra = torch.nn.LSTM(input_size_meta+input_size_ra, hidden_size, num_layers, batch_first=True).float()
        self.fc = torch.nn.Linear(hidden_size, output_size)
        self.fc_1 = torch.nn.Linear(hidden_size, 1)
        self.use_ra = use_ra
        self.device = device

    def forward(self, x, h0):
        # print(h0[0].shape)
        # print(h0[1].shape)
        (x_meta, x_ra) = x
        # print('x_meta shape:{}'.format(x_meta.shape))
        # print('x_ra shape:{}'.format(x_ra.shape))

        if self.use_ra == True:
            x_ra = x_ra.reshape(x_ra.shape[0], x_ra.shape[1], -1)
            new = torch.cat((x_ra, x_meta), 2)
            model = self.lstm_meta_ra
        else:
            new = x_meta
            model = self.lstm_meta

        _, (h, _) = model(new, h0) # x_ra:(batch_size, seq_length, input_size)
        x = self.fc(h[-1])
        return x
    

    def init_hidden(self, batch_size):
        hidden = (torch.zeros([self.n_layers, batch_size, self.hidden_size]).to(self.device), torch.zeros([self.n_layers, batch_size, self.hidden_size]).to(self.device))
        return hidden


    
class CNNTest(torch.nn.Module):
    def create_nn(self, layer_dims, flag):
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(torch.nn.Linear(layer_dims[i], layer_dims[i+1]))
            layers.append(torch.nn.ReLU())
        if flag == 1:
            layers = layers[:-1]
        return torch.nn.Sequential(*layers)
    
    def __init__(self, groups, predict_step):
        super().__init__()

        self.predict_steps = predict_step
        # self.input_dim = input_dim
        self.groups = groups

        """
        Branch_1 (7 groups):
        Conv2d: 6*11*11 -> 32*11*11
        BatchNorm2d & ReLu

        Conv2d: 32*11*11 -> 32*11*11
        BatchNorm2d & ReLu
        MaxPool2d: 32*11*11 -> 32*5*5

        fc1: 32*5*5 -> 128
        ReLu
        fc2: 128 -> 6
        ReLu
        """
        self.cov_dims = [6, 32, 32]
        layers = []
        # 8 * 11 * 11 -> 64 * 11 * 11
        layers.append(torch.nn.Conv2d(self.cov_dims[0] * groups, self.cov_dims[1] * groups, kernel_size=3, padding=1, bias=False, groups=groups))
        layers.append(torch.nn.BatchNorm2d(self.cov_dims[1] * groups))
        layers.append(torch.nn.ReLU())
        self.conv1 = torch.nn.Sequential(*layers)

        #################################
        layers = []
        # 64 * 11 * 11 -> 64 * 11 * 11
        layers.append(torch.nn.Conv2d(self.cov_dims[1] * groups, self.cov_dims[2] * groups, kernel_size=3, padding=1,
                                bias=False, groups=groups))
        layers.append(torch.nn.BatchNorm2d(self.cov_dims[2] * groups))
        layers.append(torch.nn.ReLU())
        # 64 * 11 * 11 -> 64 * 5 * 5
        layers.append(torch.nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv2 = torch.nn.Sequential(*layers)

        #################################

        self.fc_dims = [self.cov_dims[2] * 5 * 5 * groups, 128, 6]
        self.fc = self.create_nn(self.fc_dims, 0)

        # """
        # Branch_2: FFNN
        # fc1: 44 -> 128
        # ReLu
        # fc2: 128 -> 16
        # ReLu
        # """
        # self.FFNN_dims = [input_dim, 128, 16]
        # self.FFNN = self.create_nn(self.FFNN_dims, 0)


        # """
        # Fuse_NN
        # fc1: 16*6 -> 128
        # ReLu
        # fc2: 128 -> 3
        # ReLu
        # """
        # self.fuse_dims = [16*(groups) + 16, 128, predict_step*3]
        # self.fuse = self.create_nn(self.fuse_dims, 1)

    def forward(self, x):
        _, x = x
        x = x.reshape(x.shape[0], 42, 11, 11)
        x = self.conv1(x)
        x = self.conv2(x)
        # print("input shape is {}".format(x.shape))
        # x = x.view(7, -1)
        # x1 = x[1].view(-1, self.cov_dims[2] * 5 * 5)
        x = x.view(-1, 7 * self.cov_dims[2] * 5 * 5)
        # print("input shape is {}".format(x.shape))
        x = self.fc(x)

        # return a flattened 6 * 1 array of pmin
        x = x.view(-1, 6)
        # print("output shape is {}".format(x1.shape))

        # y = self.FFNN(y)
        # y = y.view(-1, 16)

        output = x

        # output = torch.cat((x, y), 1)
        # output = self.fuse(output)

        return output

class LSTM_CNN(torch.nn.Module):
    """
    para
    - input_size: feature size
    - hidden_size: number of hidden units
    - output_size: number of output --> 6 predicted timesteps
    - num_layers: layers of LSTM to stack
    """
    def __init__(self, input_size_ra, input_size_meta, hidden_size, output_size, num_layers, device, use_ra = True):
        super().__init__()
        self.input_size_ra = input_size_ra
        self.input_size_meta = input_size_meta
        self.output_size = output_size
        self.n_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm_meta = torch.nn.LSTM(input_size_meta, hidden_size, num_layers, batch_first=True).float()
        self.lstm_meta_ra = torch.nn.LSTM(input_size_meta+input_size_ra, hidden_size, num_layers, batch_first=True).float()
        self.fc = torch.nn.Linear(hidden_size, output_size)
        self.fc_1 = torch.nn.Linear(hidden_size, 1)
        self.use_ra = use_ra

        # CNN setting
        self.feature_map_sizes = [28, 14, 7]
        self.filter_sizes = [3, 3]
        self.ConvNet = ConvNet(self.feature_map_sizes, self.filter_sizes, activation=torch.nn.ReLU())  
        self.device = device

    def forward(self, x, h0):
        # print(h0[0].shape)
        # print(h0[1].shape)
        (x_meta, x_ra) = x
        # print('x_meta shape:{}'.format(x_meta.shape))
        # print('x_ra shape:{}'.format(x_ra.shape))

        if self.use_ra == True:
            cnn_concat = self.ConvNet(x_ra) # batch_size * vector_length * timesteps    ==> 56 when [28, 14, 7] & [3, 3]
            cnn_concat = cnn_concat.transpose(1, 2) # batch_size * timesteps * vector_length
            new = torch.cat((cnn_concat, x_meta), 2)
            model = self.lstm_meta_ra
        else:
            new = x_meta
            model = self.lstm_meta

        _, (h, _) = model(new, h0) # x_ra:(batch_size, seq_length, input_size)
        x = self.fc(h[-1])   # get the output of the last hidden state
        return x
    

    def init_hidden(self, batch_size):
        hidden = (torch.zeros([self.n_layers, batch_size, self.hidden_size]).to(self.device), torch.zeros([self.n_layers, batch_size, self.hidden_size]).to(self.device))
        return hidden

class GRU_CNN(torch.nn.Module):
    """
    para
    - input_size: feature size
    - hidden_size: number of hidden units
    - output_size: number of output --> 6 predicted timesteps
    - num_layers: layers of GRU to stack
    """
    def __init__(self, input_size_ra, input_size_meta, hidden_size, output_size, num_layers, device, use_ra = True):
        super().__init__()
        self.input_size_ra = input_size_ra
        self.input_size_meta = input_size_meta
        self.output_size = output_size
        self.n_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm_meta = torch.nn.GRU(input_size_meta, hidden_size, num_layers, batch_first=True).float()
        self.lstm_meta_ra = torch.nn.GRU(input_size_meta+input_size_ra, hidden_size, num_layers, batch_first=True).float()
        self.fc = torch.nn.Linear(hidden_size, output_size)
        self.use_ra = use_ra

        # CNN setting
        self.feature_map_sizes = [28, 14, 7]
        self.filter_sizes = [3, 3]
        self.ConvNet = ConvNet(self.feature_map_sizes, self.filter_sizes, activation=torch.nn.ReLU())  
        self.device = device

    def forward(self, x, h0):
        # print(h0[0].shape)
        # print(h0[1].shape)
        (x_meta, x_ra) = x
        # print('x_meta shape:{}'.format(x_meta.shape))
        # print('x_ra shape:{}'.format(x_ra.shape))

        if self.use_ra == True:
            cnn_concat = self.ConvNet(x_ra) # batch_size * vector_length * timesteps    ==> 56 when [28, 14, 7] & [3, 3]
            cnn_concat = cnn_concat.transpose(1, 2) # batch_size * timesteps * vector_length
            new = torch.cat((cnn_concat, x_meta), 2)
            model = self.lstm_meta_ra
        else:
            new = x_meta
            model = self.lstm_meta

        _, h = model(new, h0) # x_ra:(batch_size, seq_length, input_size)
        x = self.fc(h[-1])   # get the output of the last hidden state
        return x
    

    def init_hidden(self, batch_size):
        hidden = torch.zeros([self.n_layers, batch_size, self.hidden_size]).to(self.device)
        return hidden
