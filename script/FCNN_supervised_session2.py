import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from script.nnet_pytorch.model_init import model_init
from script.train_test_pytorch.train_FCNN_pytorch import training
####################################################
# Training and testing the
# supervised neural network model
####################################################
train_data = pd.read_csv('data/UJIndoorLoc/trainingData.csv', low_memory=False, index_col=0)
validation_data = pd.read_csv('data/UJIndoorLoc/validationData.csv', low_memory=False, index_col=0)

train_output = train_data['FLOOR'].to_list() # 5 levels
validation_output = validation_data['FLOOR'].to_list()

train_features = train_data.drop(["LONGITUDE", "LATITUDE", "FLOOR"], axis=1)
validation_features = validation_data.drop(["LONGITUDE", "LATITUDE", "FLOOR"], axis=1)
####################################################
batch_size = 512
epoch_num = 200
seed_shuffling = 42
torch.manual_seed(20)

net, optimizer = model_init(input_size = train_features.shape[1],
                            layers_size = [128, 64, 32],
                            output_size = len(np.unique(train_output)),
                            optimizer_params = {'method': 'Adam', 'learning_rate': 0.001, 'momentum': 0.9, 'dropout_prob': 0.4})
####################################################
network_train, layers_values, loss_dict, performance_dict = training(network_train = net,
                                                                     batch_size = batch_size,
                                                                     epochs = epoch_num,
                                                                     optimizer = optimizer,
                                                                     seed_shuffling = seed_shuffling,
                                                                     features_training = train_features,
                                                                     labels_training = train_output,
                                                                     features_validation = validation_features,
                                                                     labels_validation = validation_output,
                                                                     validation_whiletraining=True)

#################
plt.plot(loss_dict['testing_perepoch'])
plt.show()


plt.plot(performance_dict['testing_perepoch'])
plt.show()


