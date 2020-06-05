import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.autograd import Variable
####################################################
# Training and testing the
# supervised neural network model
####################################################
def training(network_train, feature_ids, chemical_feat, protein_feat,
            labels_training, batch_size, epochs,
            optimizer, seed_shuffling,
            test_set = [], testing_whiletraining=False):
   '''
   Training neural network
   :param network_train: network to be used for training
   :param feature_ids: list of feature id pairs from chemical and protein spaces matched with the training interactions
   :param chemical_feat: list of chemical features
   :param protein_feat: list of protein features
   :param labels_training: labels of trainign datapoints
   :param batch_size: batch size for training
   :param epochs: number of epochs
   :param optimizer: optimizer
   :param seed_shuffling: random seed to be used for shuffling data points for batch selection
   :param test_set: testing set
   :param testing_whiletraining: logical parameter (if True, testing is done per epoch to follow the trend of traingina and testing per epoch)
   :return: trained network, layer weights per epochs, loss and performance for training and testing
   '''

   input_featnum = len(chemical_feat[0] + protein_feat[0])
   #
   random.seed(seed_shuffling)
   indices_shufflled = list(range(len(labels_training)))
   random.shuffle(indices_shufflled)


   labels_shuffled = [labels_training[iter] for iter in indices_shufflled]
   featureids_shuffled = [feature_ids[iter] for iter in indices_shufflled]

   batch_ids = [[iter*batch_size, min(len(labels_shuffled), (iter+1)*batch_size)] for iter in range(int(len(labels_shuffled)/batch_size)+1)]

   loss_training = []
   loss_testing = []

   performance_training = []
   performance_testing = []

   layers_values = []

   criterion = nn.NLLLoss()
   for epoch in range(epochs):
       print('epoch {} '.format(epoch+1))
       #
       layerval_tmp = [network_train.layers[layer_iter].weight.tolist() for layer_iter in np.arange(0,len(network_train.layers))]

       layers_values.append(layerval_tmp)

       loss_sum = 0.0
       correct = 0.0
       for batch_idx in range(len(batch_ids)):


           features_expandlist = [np.float_(chemical_feat[featureids_shuffled[feat_iter][0]] + protein_feat[featureids_shuffled[feat_iter][1]]) for feat_iter in
                             range(batch_ids[batch_idx][0], batch_ids[batch_idx][1])]

           labels_batch = list(np.array(labels_shuffled)[range(batch_ids[batch_idx][0], batch_ids[batch_idx][1])])

           data = torch.tensor(features_expandlist, dtype=torch.float)
           target = torch.tensor(labels_batch, dtype=torch.long)

           data, target = Variable(data), Variable(target)
           data = data.view(-1, input_featnum)
           optimizer.zero_grad()
           net_out = network_train(data.float())

           loss = criterion(net_out, target)

           loss_sum += loss.item()
           loss.backward()
           optimizer.step()

           pred = net_out.data.max(1)[1]  # get the index of the max log-probability
           correct += pred.eq(target.data).sum()

       print(correct.item())
       loss_training.append(loss_sum/(np.float(len(labels_shuffled)))) #
       performance_training.append(100. * correct.item() / len(labels_training))
       print(loss_training[-1])

       # apply the model on test set
       if testing_whiletraining:
           test_performance, test_loss = testing(network_test = network_train,
                                                 features_testing = test_set['features_test'],
                                                 labels_testing = test_set['lables_test'],
                                                 batch_size = batch_size)
           loss_testing.append(test_loss)
           print('Test loss: {}'.format(test_loss))

           performance_testing.append(test_performance)

           print('Test performance: {}'.format(test_performance))
           if epoch > 0:
               print('test loss change: {}'.format((loss_testing[-1]-loss_testing[-2]) / loss_testing[-1]))
               print('test performance change: {}'.format((performance_testing[-1] - performance_testing[-2]) / performance_testing[-1]))


   loss_dict = {'training_perepoch': loss_training, 'testing_perepoch': loss_testing}
   performance_dict = {'training_perepoch': performance_training, 'testing_perepoch': performance_testing}

   return network_train, layers_values, loss_dict, performance_dict

def testing(network_test, features_testing, labels_testing, batch_size):
   '''
   Testing the trained neural network model
   :param network_test: trained network to be used for prediction
   :param features_testing: features of the test set interactions
   :param labels_testing: labels of the test set interactions
   :param batch_size: batch size
   :return: loss and accuracy of the model in the test set
   '''

   input_featnum = len(list(features_testing[0]))
   #
   batch_ids = [[iter*batch_size, min(len(labels_testing), (iter+1)*batch_size)] for iter in range(int(len(labels_testing)/batch_size)+1)]

   criterion = nn.NLLLoss()
   test_loss = 0
   correct = 0
   print('testing')
   with torch.no_grad():

       for batch_idx in range(len(batch_ids)):

           features_batch = [features_testing[feat_iter] for feat_iter in range(batch_ids[batch_idx][0], batch_ids[batch_idx][1])]
           features_expandlist = [np.float_(list(feature_list_iter)) for feature_list_iter in features_batch]

           labels_batch = list(np.array(labels_testing)[range(batch_ids[batch_idx][0], batch_ids[batch_idx][1])])#[labels_testing[iter] for iter in range(batch_ids[batch_idx][0], batch_ids[batch_idx][1])]

           data = torch.tensor(features_expandlist, dtype=torch.float)
           target = torch.tensor(labels_batch, dtype=torch.long)


           data, target = Variable(data), Variable(target)
           data = data.view(-1, input_featnum)
           net_out = network_test(data.float())

           test_loss += criterion(net_out, target).data.item()  # criterion(net_out, target).data[0]
           pred = net_out.data.max(1)[1]  # get the index of the max log-probability
           correct += pred.eq(target.data).sum()

       test_loss /= len(labels_testing)

   return 100. * correct.item() / len(labels_testing), test_loss

####################################################
# Initializing and optimizing the neural network
####################################################
def model_init(input_size, num_layers, layers_size, output_size, optimizer_params):
    '''
    Initializing the network and the optimizer
    :param input_size: number of input features
    :param num_layers: number of the hidden layers
    :param layers_size: sizes of the hidden layers
    :param output_size: number of output predictions
    :param optimizer_params: Optimization algorithm's parameters
    :return: initialized network and optimizer
    '''
    net = Net(input_size = input_size,
              num_layers = num_layers,
              layers_size = layers_size,
              output_size = output_size)

    parameters = [par for model in net.layers for par in model.parameters()]

    optimizer_init = opt.SGD(parameters, lr=optimizer_params['learning_rate'], momentum=optimizer_params['momentum'])

    return net, optimizer_init


