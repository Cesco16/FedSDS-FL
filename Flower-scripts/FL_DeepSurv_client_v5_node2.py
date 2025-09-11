# Author: Luciana Carota
# Email: luciana.carota@gmail.com
# Date: 12/02/2024

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

warnings.filterwarnings("ignore")

n_splits = 10

from sklearn.model_selection import KFold

cv = KFold(n_splits=n_splits, random_state=1, shuffle=True)

######### data load ####################


get_target = lambda df: (df.iloc[:, 0].values.astype('int32'), df.iloc[:, 1].values.astype('int32'))

# data_path_train='./Scenarios_normalized_12mut/Scenario_1_normalized_12mut.xlsx'
# data_path_train='/home/dpiscia/mds_datasets/scenario_1_node_1_data.csv'

data_path_train = './Node2/MDS/'  # MDS_5NODES/'+node_name+'/'+scenario1+'/df_'+scenario2+'.csv'
##data_path_train = '/home/PERSONALE/francesco.casadei20/FL_simul/MDS_5NODES/'+node_name+'/'+scenario1+'/df_'+scenario2+'.csv'

##file_name = './DeepSurv_cv_val_MDS_platform_1nodes_v2.pk'
file_name = './DeepSurv_cv_mds.pk'
##file_name = './DS_uniquetest_MDS_platform_v1_noinit.pk'
##file_name = './DeepSurv_cv_TEST_MDS_platform_v2.pk'
with open(file_name, 'rb') as f:
    # model = pickle.load(f).create_model(in_features, encoded_features, out_features,trainloader, testloader, testloader_eval)
    # model = pickle.load(f).create_model()

    model = pickle.load(f).create_model(dataset_path=data_path_train)
    # (x_train, y_train), (x_test, y_test) = model.data
    print('load data in client.py')
    (x_train, y_train), (x_test, y_test) = model.data


# (x_train, y_train), (x_test, y_test) = model.load_data(data_path_train)


class NumpyClientWrapper(fl.client.NumPyClient):

    def __init__(self):
        self.central_epoch = 0
        self.local_epoch = 0

    def get_parameters(self, config):
        return model.get_weight()

    def fit(self, parameters, config):
        model.set_weight(parameters)
        model.fit(x_train, y_train, epochs=1, batch_size=512, steps_per_epoch=1)
        return model.get_weight(), len(x_train), {}

    def evaluate(self, parameters, config):
        print('EVALUATE CLIENT')
        model.set_weight(parameters)
        loss, accuracy = model.evaluate(x_train, y_train)
        self.central_epoch = self.central_epoch + 1

        return loss, len(x_train), {"accuracy": accuracy}


client = NumpyClientWrapper()
# Start Flower client
print('Start flower client')
# fl.client.start_numpy_client(server_address=str('137.204.51.140:8437'), client=client)
fl.client.start_numpy_client(server_address=str('172.17.34.44:8005'), client=client)


