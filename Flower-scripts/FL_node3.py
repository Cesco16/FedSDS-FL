# Author: Francesco Casadei, Luciana Carota
# Email: francesco.casadei20@unibo.it, luciana.carota@unibo.it
# Date: 19/05/2025

# Import libraries
import os
import pickle
import warnings

import flwr as fl
import numpy as np
import pandas as pd
import torchtuples as tt

from typing import Dict
from typing import Tuple, Union, List
from collections import OrderedDict

from sklearn.model_selection import train_test_split
n_splits=10
connection_port = '8005'
ip = '000.00.00.00'


from sklearn.model_selection import KFold
cv = KFold(n_splits=n_splits, random_state=1, shuffle=True)

######### data load ####################  

get_target = lambda df: (df.iloc[:,0].values.astype('int32'), df.iloc[:,1].values.astype('int32'))

data_path_train = './Node3/MDS/'
file_name = './DeepSurv_cv_mds.pk'

with open(file_name, 'rb') as f:

    model = pickle.load(f).create_model(data_path_train)
    print('load data in client.py')
    (x_train, y_train), (x_test, y_test) = model.data

class NumpyClientWrapper(fl.client.NumPyClient):

    def __init__(self):
            self.central_epoch = 0
            self.local_epoch = 0

    def get_parameters(self,config):
        return model.get_weight()

    def fit(self, parameters, config):
        model.set_weight(parameters)
        model.fit(x_train, y_train, epochs=1, batch_size=512, steps_per_epoch=1)
        return model.get_weight(), len(x_train), {}

    def evaluate(self, parameters, config):
        print('EVALUATE CLIENT')
        model.set_weight(parameters)
        loss, accuracy = model.evaluate(x_train, y_train)
        self.central_epoch=self.central_epoch+1
    
        return loss, len(x_train), {"accuracy": accuracy}


client = NumpyClientWrapper()

# Start Flower client

print('Start flower client')
fl.client.start_numpy_client(server_address=str(ip+':'+connection_port), client=client)
