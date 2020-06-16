import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as opt
from torch import nn
####################################################








####################################################
# Training and the variational autoencoder
####################################################
def training(network_train, features, batch_size, epochs, optimizer):
   '''
   Training neural network
   :param network_train: network to be trained
   :param features: list of features to be used in variational autoencoder
   :param batch_size: batch size
   :param epochs: number of epochs
   :param optimizer: optimizer
   :return: trained network, layer weights and loss per epoch
   '''
   indices_shufflled = list(range(len(features)))

   featureids_shuffled = [iter for iter in indices_shufflled]
   batch_ids = [[iter*batch_size, min(len(features), (iter+1)*batch_size)] for iter in range(int(len(features)/batch_size)+1)]

   loss_training = []
   layers_values = []
   for epoch in range(epochs):
       print('epoch {} '.format(epoch+1))
       #
       layerval_tmp = [network_train.layers[layer_iter].weight.tolist() for layer_iter in np.arange(0,len(network_train.layers))]
       layers_values.append(layerval_tmp)

       loss_sum = 0.0
       for batch_idx in range(len(batch_ids)):
           features_expandlist = [np.float_(features[featureids_shuffled[feat_iter]]) for feat_iter in range(batch_ids[batch_idx][0], batch_ids[batch_idx][1])]

           data = torch.tensor(features_expandlist, dtype=torch.float) #torch.tensor(features_expandlist, dtype=torch.bool)
           data= Variable(data)
           data = data.view(data.size(0), -1)

           optimizer.zero_grad()
           recon_batch, mu, logvar = network_train(data.float())

           loss = loss_function(recon_batch, data, mu, logvar)
           loss_sum += loss.item()

           loss.backward()
           optimizer.step()

       loss_training.append(loss_sum/len(features))
       print(np.log10(loss_training[-1]))
       if len(loss_training) > 2:
           print(1000.0*abs((loss_training[-1]-loss_training[-2])/loss_training[-1]))

   return network_train, layers_values, loss_training

####################################################
# Initializing and optimizing the neural network
####################################################
class VAE(nn.Module):

   def __init__(self, input_size, layers_size):
       '''
       Function for building the network
       :param input_size: number of input features
       :param layers_size: sizes of the hidden layers
       '''
       super(VAE, self).__init__()

       layers_decoder = layers_size[::-1]
       self.layers_size = layers_size
       self.input_size = input_size
       self.layers_decoder = layers_decoder

       self.layers = [nn.Linear(input_size, layers_size[0])]
       for i in range(1,len(layers_size)):
           self.layers.append(nn.Linear(layers_size[i-1], layers_size[i]))
       #
       self.layers.append(nn.Linear(layers_size[len(layers_size)-2], layers_size[len(layers_size)-1]))
       #
       for i in range(1,len(layers_decoder)):
           self.layers.append(nn.Linear(layers_decoder[i-1], layers_decoder[i]))
       #
       self.layers.append(nn.Linear(layers_decoder[-1], input_size))

   def encode(self ,x):
       for i in range(0, len(self.layers_size) - 1):
           x = F.relu(self.layers[i](x))
       mu = self.layers[len(self.layers_size)-1](x)
       logvar = self.layers[len(self.layers_size)](x)
       return mu, logvar

   def decode(self ,z):
       for i in range(len(self.layers_size)+1, len(self.layers) - 1):
           z = F.relu(self.layers[i](z))
       out = torch.sigmoid(self.layers[len(self.layers)-1](z))
       return out

   def reparameterize(self, mu, logvar):
       std = torch.exp(0.5 * logvar)
       eps = torch.randn_like(std)
       return eps.mul(std).add_(mu)

   def forward(self ,x):
       mu, logvar = self.encode(x)
       z = self.reparameterize(mu, logvar)
       decode_z = self.decode(z)
       return decode_z, mu, logvar

def model_init(input_size, layers_size, optimizer_params): #learning_rate, momentum
   '''
   Function for initializing the network and the optimizer
   :param input_size: number of input features
   :param num_layers: number of the hidden layers
   :param layers_size: sizes of the hidden layers
   :param output_size: number of output predictions
   :param optimizer_params: Optimization algorithm's parameters
   :return: initialized network and optimizer
   '''

   net = VAE(input_size = input_size,
             layers_size = layers_size)

   parameters = [par for model in net.layers for par in model.parameters()]

   if optimizer_params['method'] == 'SGD':
       optimizer_init = opt.SGD(parameters, lr=optimizer_params['learning_rate'], momentum = optimizer_params['momentum'])
   elif optimizer_params['method'] == 'Addelta':
       optimizer_init = opt.Adadelta(parameters, lr=optimizer_params['learning_rate'])
   elif optimizer_params['method'] == 'Adam':
       optimizer_init = opt.Adam(parameters, lr=optimizer_params['learning_rate'])

   return net, optimizer_init

def loss_function(recon_x, x, mu, logvar):
   BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
   KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

   return BCE + KLD
####################################################
# Generating embeddings of the
# target space using the trained model
####################################################
def latent_extract(network, features):
   '''
   Extrating embeddings
   :param network: trained neural network
   :param features: list of features
   :return: list of embeddings
   '''
   latent_list = []
   for feat_iter in range(0, len(features)):
       # print(feat_iter)
       data = torch.tensor([np.float_(features[feat_iter])], dtype=torch.float)  # torch.tensor(features_expandlist, dtype=torch.bool)
       data = Variable(data)
       data = data.view(data.size(0), -1)
       recon_batch, mu, logvar = network(data.float())
       std = torch.exp(0.5 * logvar)
       eps_val = torch.full_like(mu, fill_value=0 * 0.01)
       z_val = eps_val.mul(std).add_(mu)
       latent_list.append(np.array(z_val[0].data))

   return latent_list
####################################################
# Visualizing loss and saving the trained model
####################################################
def VSE_loss_plot(target_space, loss_training, epoch_num):
   '''
   Plottin the loss and loss change per epoch while training the VAE
   :param target_space: name of target space
   :param loss_training: list of loss per epoch
   :param epoch_num: number of epochs
   '''
   plt.plot(np.arange(5, len(loss_training)),loss_training[5:len(loss_training)])
   plt.xticks(fontsize=14)
   plt.yticks(fontsize=14)
   plt.ylabel('loss', fontsize = 14)
   plt.xlabel('epochs', fontsize = 14)
   plt.savefig('figures/VAE_' + target_space + '_' + str(epoch_num) + 'epochs_loss.png')
   plt.show()

   normslope = [abs(loss_training[iter] - loss_training[(iter - 1)]) / loss_training[iter] for iter in
                range(5, len(loss_training))]
   plt.plot(np.arange(5, len(loss_training)), normslope)
   plt.xticks(fontsize=14)
   plt.yticks(fontsize=14)
   plt.ylabel('normalized loss change', fontsize = 14)
   plt.xlabel('epochs', fontsize = 14)
   plt.savefig('figures/VAE_' + target_space + '_' + str(epoch_num) + 'epochs_loss_normslope.png')
   plt.show()


