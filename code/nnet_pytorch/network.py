import torch
import torch.nn as nn
import torch.nn.functional as F
##########
class FCNN(nn.Module):

    def __init__(self, input_size, num_layers, layers_size, output_size):
       '''
       Function for building the network
       :param input_size: number of input features
       :param num_layers: number of the hidden layers
       :param layers_size: sizes of the hidden layers
       :param output_size: number of output predictions
       '''
       super(FCNN, self).__init__()

       self.FC1 = nn.Linear(input_size, layers_size[0])
       self.FC2 = nn.Linear(layers_size[0], layers_size[1])
       self.FC3 = nn.Linear(layers_size[1], layers_size[-1])
       self.FC4 = nn.Linear(layers_size[-1], output_size)

   ###########################################
   # Forward passing in the network
   # implementation of activation
   # functions in each layer
   ###########################################
   def forward(self, x):
       x = F.relu(self.FC1(x))
       x = F.relu(self.FC2(x))
       x = F.relu(self.FC3(x))
       x = F.relu(self.FC4(x))

       return F.log_softmax(x, dim = 1)