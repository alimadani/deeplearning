import torch.optim as opt
from script.nnet_pytorch.network import FCNN
####

def model_init(input_size, layers_size, output_size, optimizer_params):
    '''
    Initializing the network and the optimizer
    :param input_size: number of input features
    :param layers_size: sizes of the hidden layers
    :param output_size: number of output predictions
    :param optimizer_params: Optimization algorithm's parameters
    :return: initialized network and optimizer
    '''
    net = FCNN(input_size = input_size,
               layers_size = layers_size,
               output_size = output_size,
               p = optimizer_params['dropout_prob'])

    parameters = [par for model in net.layers for par in model.parameters()]

    if optimizer_params['method'] == 'SGD':
        optimizer_init = opt.SGD(parameters, lr=optimizer_params['learning_rate'], momentum = optimizer_params['momentum'])
    elif optimizer_params['method'] == 'Addelta':
        optimizer_init = opt.Adadelta(parameters, lr=optimizer_params['learning_rate'])
    elif optimizer_params['method'] == 'Adam':
        optimizer_init = opt.Adam(parameters, lr=optimizer_params['learning_rate'])

    return net, optimizer_init