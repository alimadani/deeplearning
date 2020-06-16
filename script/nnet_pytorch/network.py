import torch.nn as nn
import torch.nn.functional as F
##########
class FCNN(nn.Module):

    def __init__(self, input_size, layers_size, output_size, p = 0.2):
        '''
        Function for building the network
        :param input_size: number of input features
        :param layers_size: sizes of the hidden layers
        :param p: probability of dropping neurons in dropout
        :param output_size: number of output predictions
        '''
        super(FCNN, self).__init__()
        self.p = p
        self.layers = [nn.Linear(input_size, layers_size[0])]
        self.layers.append(nn.BatchNorm1d(num_features=layers_size[0]))
        for i in range(1,len(layers_size)):
            self.layers.append(nn.Linear(layers_size[i-1], layers_size[i]))
            self.layers.append(nn.BatchNorm1d(num_features=layers_size[i]))
        self.layers.append(nn.Linear(layers_size[-1], output_size))
    ###########################################
    # Forward passing in the network
    # implementation of activation
    # functions in each layer
    ###########################################
    def forward(self, x):
        for i in range(0, len(self.layers) - 1, 2):
            x_tmp = F.relu(F.dropout(self.layers[i](x), p=self.p, training=True))
            x = self.layers[i + 1](x_tmp)
        x = self.layers[-1](x)
        return F.log_softmax(x, dim = 1)