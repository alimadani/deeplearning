import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
################
def training(network_train, batch_size, epochs,
             optimizer, seed_shuffling,
             features_training, labels_training,
             features_validation = pd.DataFrame(),
             labels_validation = pd.DataFrame(),
             validation_whiletraining=False):
   '''
   Training neural network
   :param network_train: network to be used for training
   :param features_training: dataframe of features for training set
   :param labels_training: labels of training datapoints
   :param features_validation: dataframe of features for validation set
   :param labels_validation: labels of validation set
   :param batch_size: batch size for training
   :param epochs: number of epochs
   :param optimizer: optimizer
   :param seed_shuffling: random seed to be used for shuffling data points for batch selection
   :param testing_whiletraining: logical parameter (if True, testing is done per epoch to follow the trend of training and testing per epoch)
   :return: trained network, layer weights per epochs, loss and performance for training and testing
   '''

   input_featnum = features_training.shape[1]
   #
   random.seed(seed_shuffling)
   indices_shufflled = list(range(len(labels_training)))
   random.shuffle(indices_shufflled)


   labels_shuffled = [labels_training[iter] for iter in indices_shufflled]
   featureids_shuffled = [range(0, features_training.shape[0])[iter] for iter in indices_shufflled]

   batch_ids = [[iter*batch_size, min(len(labels_shuffled), (iter+1)*batch_size)]
                for iter in range(int(len(labels_shuffled)/batch_size)+1)]


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


           features_list = [np.float_(features_training.iloc[featureids_shuffled[feat_iter],:]) for feat_iter in
                             range(batch_ids[batch_idx][0], batch_ids[batch_idx][1])]
           labels_batch = list(np.array(labels_shuffled)[range(batch_ids[batch_idx][0], batch_ids[batch_idx][1])])

           data = torch.tensor(features_list, dtype=torch.float)
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

       print('Number of correct items in training : {}'.format(correct.item()))
       loss_training.append(loss_sum/(np.float(len(labels_shuffled)))) #
       performance_training.append(100. * correct.item() / len(labels_training))
       print('Training loss: {}'.format(loss_training[-1]))

       # apply the model on test set
       if validation_whiletraining:
           test_performance, test_loss = testing(network_test = network_train,
                                                 features_validation = features_validation,
                                                 labels_validation = labels_validation,
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

def testing(network_test, features_validation, labels_validation, batch_size):
   '''
   Testing the trained neural network model
   :param network_test: trained network to be used for prediction
   :param features_testing: dataframe of features for validation set
   :param labels_testing: labels of validation set
   :param batch_size: batch size
   :return: loss and accuracy of the model in the test set
   '''

   input_featnum = features_validation.shape[1]
   #
   batch_ids = [[iter*batch_size, min(len(labels_validation), (iter+1)*batch_size)]
                for iter in range(int(len(labels_validation)/batch_size)+1)]
   criterion = nn.NLLLoss()
   test_loss = 0
   correct = 0
   with torch.no_grad():

       for batch_idx in range(len(batch_ids)):

           features_list = [np.float_(features_validation.iloc[feat_iter,:]) for feat_iter in
                             range(batch_ids[batch_idx][0], batch_ids[batch_idx][1])]
           # features_batch = [features_validation[feat_iter] for feat_iter in range(batch_ids[batch_idx][0], batch_ids[batch_idx][1])]
           # features_list = [np.float_(list(feature_list_iter)) for feature_list_iter in features_batch]

           labels_batch = list(np.array(labels_validation)[range(batch_ids[batch_idx][0], batch_ids[batch_idx][1])])

           data = torch.tensor(features_list, dtype=torch.float)
           target = torch.tensor(labels_batch, dtype=torch.long)


           data, target = Variable(data), Variable(target)
           data = data.view(-1, input_featnum)
           net_out = network_test(data.float())

           test_loss += criterion(net_out, target).data.item()  # criterion(net_out, target).data[0]
           pred = net_out.data.max(1)[1]  # get the index of the max log-probability
           correct += pred.eq(target.data).sum()

       test_loss /= len(labels_validation)

   return 100. * correct.item() / len(labels_validation), test_loss
