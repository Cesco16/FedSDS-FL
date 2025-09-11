# Author: Francesco Casadei
# Email: francesco.casadei20@unibo.it
# Date: 19/02/2023
# Modified by Luciana Carota for a unique test file and 5 nodes 22/11/2024

# MODEL DEEPSURV FROM PYCOX

# Import libraries
import mlflow
from collections import OrderedDict
from typing import Tuple, Dict
import os
import warnings

warnings.filterwarnings("ignore")

def load_data(data_path_train) -> Tuple[Dict, Dict]:
    import pandas as pd
    from os.path import realpath
    from os.path import dirname
    from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder

    # all the features
    continuous = ['age_at_index', 'age_at_diagnosis']
    binary = ['gender', 'prior_malignancy', 'prior_treatment']
    categorical = ['race', 'morphology', 'primary_diagnosis', 'year_of_diagnosis', 'tissue_or_organ_of_origin','treatment_or_therapy',]
                   #'ajcc_pathologic_m', 'ajcc_pathologic_n', 'ajcc_pathologic_stage', 'ajcc_pathologic_t','ajcc_staging_system_edition']
    survs = ['time', 'event']

    features = continuous + categorical + binary + survs

    #function to prepare the y variable
    get_target = lambda df: (df.iloc[:, 0].values.astype('int32'), df.iloc[:, 1].values.astype('int32'))
    #data_path_train = './Node1/'

    #Read data
    # SYNTHETIC UNBIASED
    #g1 = pd.read_csv(data_path_train + 'synth_brca1__unbiased_denorm.csv')[features].reset_index(drop=True)
    #g2 = pd.read_csv(data_path_train + 'synth_brca2__unbiased_denorm.csv')[features].reset_index(drop=True)
    #G = pd.concat((g1, g2), axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
    # SYNTHETIC BIASED
    #s11 = pd.read_csv(data_path_train + 'brca1__biased_denorm_brca1_.csv', index_col=0)[features].reset_index(drop=True)#.loc[:frac1]  # .sample(frac=0.6, random_state=42) #300
    #s12 = pd.read_csv(data_path_train + 'brca2__biased_denorm_brca1_.csv', index_col=0)[features].reset_index(drop=True)#.loc[:frac2]

    #S1 = pd.concat((s11, s12), axis=0)
    # REAL DATA
    r1 = pd.read_csv(data_path_train + 'lgg_.csv')
    # TEST DATA
    t1 = pd.read_csv(data_path_train + 'testlgg_.csv')

    #ORDER COLUMNS
    r1 = r1[features]
    t1 = t1[features]

    # for biased mode
    #N1 = pd.concat((r1, S1), axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
    # for unbiased mode
    #G1 = pd.concat((r1, G), axis=0).sample(frac=1, random_state=11).reset_index(drop=True)

    N1 = pd.read_csv(data_path_train + 'SYNTH_BIASED.csv', index_col=0, sep=',')
    #print(N1)
    N1 = N1[features].sample(frac=1, random_state=42).reset_index(drop=True).copy()
    G1 = pd.read_csv(data_path_train + 'SYNTH_UNBIASED.csv', index_col=0, sep=',')
    G1 = G1[features].sample(frac=1, random_state=42).reset_index(drop=True).copy()
    R = pd.read_csv(data_path_train + 'CENTRALIZED.csv', index_col=0, sep=',')
    R = R[features].sample(frac=1, random_state=42).reset_index(drop=True).copy()

    mode = 'centralized'

    if mode == 'fed_bias':
        train1 = N1.copy()
    elif mode == 'iso' or mode == 'fed_avg':
        train1 = r1.copy()
    elif mode == 'fed_unbias':
        train1 = G1.copy()
    elif mode == 'centralized':
        train1 = R.copy()

    test1 = t1.copy()

    train_surv1 = train1[survs].astype('int64')
    train_surv1 = pd.DataFrame(train_surv1)
    train_surv1 = get_target(train_surv1)
    for i in survs:
        features.remove(i)
    test_surv1 = test1[survs].astype('int64')
    test_surv1 = pd.DataFrame(test_surv1)
    test_surv1 = get_target(test_surv1)

    train1 = train1[features].astype('float32').copy()
    test1 = test1[features].astype('float32').copy()

    # scale the training set
    train1 = train1[features].astype('float32')
    scalerv1 = MinMaxScaler()
    train1[continuous] = scalerv1.fit_transform(train1[continuous]).astype('float32')

    # scale the test set
    test1 = test1[features].astype('float32')
    scalert1 = MinMaxScaler()  # StandardScaler()#
    test1[continuous] = scalert1.fit_transform(test1[continuous]).astype('float32')

    #encode the categorical variables in the training set

    categories = [
        [0, 1, 2, 3], #race
        [0, 1, 2, 3, 4], #morphology
        [0, 1, 2, 3, 4], #primary_diagnosis
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], #year_of_diagnosis
        [0, 1, 2, 3], #tissue_or_organ_of_origin
        [0, 1, 2] #treatment_or_therapy
    ]

    encoder = OneHotEncoder(categories=categories, handle_unknown='ignore', sparse=False)
    X = train1[categorical]
    X_encoded = encoder.fit_transform(X)
    # Ottieni i nomi delle nuove colonne
    encoded_columns = encoder.get_feature_names_out(categorical)
    # Ricostruiamo un DataFrame
    df_encoded = pd.DataFrame(X_encoded, columns=encoded_columns).astype('float32')
    # Rimuovi le colonne categoriche originali dal DataFrame
    df_no_cat = train1.drop(columns=categorical)
    # Unisci DataFrame numerico con quello codificato
    train1 = pd.concat([df_no_cat, df_encoded], axis=1)

    # encode the categorical variables in the test set

    categories = [
        [0, 1, 2, 3],  # race
        [0, 1, 2, 3, 4],  # morphology
        [0, 1, 2, 3, 4],  # primary_diagnosis
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],  # year_of_diagnosis
        [0, 1, 2, 3],  # tissue_or_organ_of_origin
        [0, 1, 2]  # treatment_or_therapy
    ]

    encoder = OneHotEncoder(categories=categories, handle_unknown='ignore', sparse=False)
    X = test1[categorical]
    X_encoded = encoder.fit_transform(X)
    # Ottieni i nomi delle nuove colonne
    encoded_columns = encoder.get_feature_names_out(categorical)
    # Ricostruiamo un DataFrame
    df_encoded = pd.DataFrame(X_encoded, columns=encoded_columns).astype('float32')
    # Rimuovi le colonne categoriche originali dal DataFrame
    df_no_cat = test1.drop(columns=categorical)
    # Unisci DataFrame numerico con quello codificato
    test1 = pd.concat([df_no_cat, df_encoded], axis=1)

    # create the dataloader
    xtrain = train1
    ytrain = (train_surv1[0], train_surv1[1])
    xtest = test1
    scaler1 = MinMaxScaler()
    el1 = scaler1.fit_transform(test_surv1[0].reshape(-1, 1))
    ytest = (el1.flatten(), test_surv1[1])
    #trainloader1 = (np.array(xtrain1), ytrain1)
    #testloader1 = (np.array(xtest1), ytest1)

    # subdivision of train in 10 subgroups for cross-validation
    n_split = 10
    sss = StratifiedShuffleSplit(n_splits=n_split, test_size=0.05, random_state=11)

    keys_train = ['train0', 'train1', 'train2', 'train3', 'train4', 'train5', 'train6', 'train7', 'train8', 'train9']
    keys_val = ['val0', 'val1', 'val2', 'val3', 'val4', 'val5', 'val6', 'val7', 'val8', 'val9']
    keys_test = ['test0', 'test1', 'test2', 'test3', 'test4', 'test5', 'test6', 'test7', 'test8', 'test9']
    # initialize empty dictionary
    Xtrain = {}
    Ytrain = {}
    Xtest = {}
    Ytest = {}
    from copy import deepcopy
    # ytrain_sub = deepcopy(ytrain)

    for i, (train_index, test_index) in enumerate(sss.split(xtrain, ytrain[1])):
        import numpy as np

        xtrain_sub = xtrain.iloc[train_index, :]
        # ytrain_sub = ytrain.iloc[train_index]
        ytrain_sub_0 = pd.DataFrame(ytrain[0][train_index]).reset_index(drop=True)
        ytrain_sub_1 = pd.DataFrame(ytrain[1][train_index]).reset_index(drop=True)
        ytrain_sub = pd.concat((ytrain_sub_0, ytrain_sub_1), axis=1)
        xtrain_sub = xtrain_sub.reset_index(drop=True)
        ytrain_sub = get_target(ytrain_sub)

        Xtrain[keys_train[i]] = xtrain_sub
        Ytrain[keys_train[i]] = ytrain_sub

        xtest_sub = xtrain.iloc[test_index, :].reset_index(drop=True)
        ytest_sub_0 = pd.DataFrame(ytrain[0][test_index]).reset_index(drop=True)
        ytest_sub_1 = pd.DataFrame(ytrain[1][test_index]).reset_index(drop=True)
        ytest_sub = pd.concat((ytest_sub_0, ytest_sub_1), axis=1)
        ytest_sub = get_target(ytest_sub)

        ##Xtest[keys[i]] = xtest_sub
        Xtrain[keys_val[i]] = xtest_sub
        Ytrain[keys_val[i]] = ytest_sub
        Xtrain[keys_test[i]] = xtest
        Ytrain[keys_test[i]] = ytest
        Xtest[keys_test[i]] = xtest
        Ytest[keys_test[i]] = ytest
        # Ytest[keys[i]] = ytest_sub

    #print('return Xtrain, Ytrain')
    print('Train in ', data_path_train, ': ', train1.shape)
    print('Test in ', data_path_train, ': ', test1.shape)
    return (Xtrain, Ytrain), (Xtest, Ytest)



class FLModel(mlflow.pyfunc.PythonModel):
    def create_strategy(self, num_rounds=100):
        import io
        import zipfile

        def create_zip_in_memory(files):
            # Create an in-memory byte stream
            in_memory_zip = io.BytesIO()

            # Create a ZipFile object with the in-memory byte stream
            with zipfile.ZipFile(in_memory_zip, mode='w') as zf:
                # Iterate over files and add them to the zip file
                for filename, file_content in files.items():
                    zf.writestr(filename, file_content)

            # Reset the byte stream's file position to the beginning
            in_memory_zip.seek(0)

            # Return the in-memory byte stream containing the zip file data
            return in_memory_zip

        import flwr
        import numpy as np
        from flwr.server.client_proxy import ClientProxy
        from typing import List, Tuple, Union, Optional, Dict
        from flwr.common import FitRes, Scalar, Parameters, parameters_to_ndarrays

        class AggregateCustomMetricStrategy(flwr.server.strategy.FedAvg):
            def aggregate_fit(self,
                              server_round: int,
                              results: List[Tuple[ClientProxy, FitRes]],
                              failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
                              ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
                # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
                aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

                if aggregated_parameters is not None and server_round == num_rounds:
                    # Convert `Parameters` to `List[np.ndarray]`
                    aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(aggregated_parameters)

                    # Save aggregated_ndarrays
                    # print(f"Saving round {server_round} aggregated_ndarrays...")
                    # np.savez(f"https://urldefense.proofpoint.com/v2/url?u=http-3A__results.zip&d=DwIGAg&c=5rLNXN0mp_7LMh3Fds96xpjyD06ZuE2RU7zikolS0lg&r=InnoW2J7WO2GyBp1EYkPTuFH7X5gCTIwBjlI02urYgc&m=txI-RX7PkACrYpay7SwX5NXUz0PjXpxO0i1t9mV9tBsiw6ZBmJaRCLwAFTCnWewe&s=1s9SXyY6Q47h_-q6hZPb8CayrsjANrGTAOwYxjuwF54&e= ", *aggregated_ndarrays)
                    # print(f"Saving zip{server_round} aggregated_ndarrays...")

                    # files = {
                    #     'file1.txt': b'Content of file 1',  # TODO: save something in files (c-index during epochs)
                    #     'file2.txt': b'Content of file 2'
                    # }

                    # # Create the zip file in memory
                    # in_memory_zip = create_zip_in_memory(files)
                    # with open('https://urldefense.proofpoint.com/v2/url?u=http-3A__results.zip&d=DwIGAg&c=5rLNXN0mp_7LMh3Fds96xpjyD06ZuE2RU7zikolS0lg&r=InnoW2J7WO2GyBp1EYkPTuFH7X5gCTIwBjlI02urYgc&m=txI-RX7PkACrYpay7SwX5NXUz0PjXpxO0i1t9mV9tBsiw6ZBmJaRCLwAFTCnWewe&s=1s9SXyY6Q47h_-q6hZPb8CayrsjANrGTAOwYxjuwF54&e= ', 'wb') as f:
                    #     f.write(in_memory_zip.getvalue())

                return aggregated_parameters, aggregated_metrics

        return AggregateCustomMetricStrategy(min_available_clients=1,
                                             min_evaluate_clients=1,
                                             min_fit_clients=1,
                                             fraction_fit=1.0,
                                             fraction_evaluate = 1.0
        )

    def create_model(self, dataset_path=""):
        import torch  # For building the networks
        import numpy as np
        import pandas as pd
        from torch import nn
        from typing import Tuple
        import torchtuples as tt  # Some useful functions
        import math as math
        import os
        import warnings
        from pycox import models
        import psutil
        import random

        from pycox.models import CoxPH
        from pycox.evaluation import EvalSurv
        from pycox.models.loss import CoxPHLoss

        from os.path import realpath
        from os.path import dirname

        torch.manual_seed(1235)

        seed = 1235
        np.random.seed(seed)

        params = {
            'in_features': 0,  # dinamically created in PhytonWrapper
            'num_nodes': [32,32],#[64,64,64],#[32, 32],
            'out_features': 1,
            'batch_norm': True,  # False,
            'dropout': 0.5,#0.5,#0.7,
            'output_bias': False,
            'activation': torch.nn.SELU,#torch.nn.ReLU,
            'w_decay': 0,
            'batch_size': 256,#512,  # 126
            'num_epochs_int': 10,
            'max_epochs': 100,
            'lr': 0.001,#0.01
        }

        def obtain_c_index(surv_f, time, censor):
            # Evaluate using PyCox c-index
            ev = EvalSurv(surv_f, time.flatten(), censor.flatten(), censor_surv="km")
            ci = ev.concordance_td()

            # Obtain also ibs
            #time_grid = np.linspace(time.min(), time.max(), 100)
            # ibs = ev.integrated_brier_score(time_grid)
            return ci

        def FileSave(filename, content):  # to remove in real test
            with open(filename, "a") as myfile:
                myfile.write(content)

        class PythonModelWrapper:
            def __init__(self, dataset_path):
                # def __init__(self, model):
                self.data = self.load_data(dataset_path)
                # print(self.data)
                # print(self.data[0])
                # print(len(self.data[0]['train0'].columns))
                # in_features=len(self.data[0]['train0'].columns)#[0]

                in_features = len(self.data[0][0]['train0'].columns)
                # print('in features')
                params['in_features'] = in_features

                self.patients_train = len(self.data[0][0]['train0'])
                self.patients_test = len(self.data[0][0]['test0'])

                seed = 1235

                random.seed(seed)
                os.environ['PYTHONHASHSEED'] = str(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                _ = torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                net = tt.practical.MLPVanilla(params['in_features'], num_nodes=params['num_nodes'],
                                              out_features=params['out_features'],
                                              batch_norm=params['batch_norm'], dropout=params['dropout'],
                                              output_bias=params['output_bias'],
                                              activation=params['activation'])

                data_path = realpath(dirname(dataset_path))
                data_path_cindex = data_path + "\\cindex.csv"

                self.model = CoxPH(net, tt.optim.Adam(weight_decay=params['w_decay']))
                self.model.net.to(device)

                #print('weights: ', [val.cpu().numpy() for _, val in self.model.net.state_dict().items()])

                # self.model = model
                self.cindex = []
                self.training_stats = {}
                self.central_epoch = 0
                self.metrics_file = data_path_cindex

            def predict(self, model_input):
                return self.model.predict(model_input)

            def fit(self, X_train, Y_train, epochs=100, batch_size=params['batch_size'], steps_per_epoch=1):

                from sklearn.preprocessing import MinMaxScaler, StandardScaler
                from sklearn.model_selection import train_test_split

                num_epochs_int = params['num_epochs_int']
                batch_size = params['batch_size']
                max_epochs = params['max_epochs']

                print('\n Try to fit ...')
                training_stats = {'c_index': [], 'ibs': [], 'epochs': [], 'num_round': [],
                                  'node_name': []}  # To store training data
                # print(X_train.keys())

                #print('MAIN EPOCH: ', self.central_epoch)

                if (self.central_epoch % max_epochs) == 0:
                    print('NEW CV!')
                    seed = 1235

                    random.seed(seed)
                    os.environ['PYTHONHASHSEED'] = str(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    _ = torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False

                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    net = tt.practical.MLPVanilla(params['in_features'], num_nodes=params['num_nodes'],
                                                  out_features=params['out_features'],
                                                  batch_norm=params['batch_norm'], dropout=params['dropout'],
                                                  output_bias=params['output_bias'],
                                                  activation=params['activation'])
                    self.model = CoxPH(net, tt.optim.Adam(weight_decay=params['w_decay']))

                    self.model.net.to(device)
                    self.model.optimizer.set_lr(params['lr'])

                    #print('weights: ', [val.cpu().numpy() for _, val in self.model.net.state_dict().items()])

                if self.central_epoch < max_epochs:
                    x_train = X_train["train0"]
                    y_train = Y_train["train0"]
                    x_val = X_train["val0"]
                    y_val = Y_train["val0"]
                    x_test = X_train["test0"]
                    y_test = Y_train["test0"]
                elif (self.central_epoch >= max_epochs) and (self.central_epoch < 2 * max_epochs):
                    x_train = X_train["train1"]
                    y_train = Y_train["train1"]
                    x_val = X_train["val1"]
                    y_val = Y_train["val1"]
                    x_test = X_train["test1"]
                    y_test = Y_train["test1"]
                elif (self.central_epoch >= 2 * max_epochs) and (self.central_epoch < 3 * max_epochs):
                    x_train = X_train["train2"]
                    y_train = Y_train["train2"]
                    x_val = X_train["val2"]
                    y_val = Y_train["val2"]
                    x_test = X_train["test2"]
                    y_test = Y_train["test2"]
                elif (self.central_epoch >= 3 * max_epochs) and (self.central_epoch < 4 * max_epochs):
                    x_train = X_train["train3"]
                    y_train = Y_train["train3"]
                    x_val = X_train["val3"]
                    y_val = Y_train["val3"]
                    x_test = X_train["test3"]
                    y_test = Y_train["test3"]
                elif (self.central_epoch >= 4 * max_epochs) and (self.central_epoch < 5 * max_epochs):
                    x_train = X_train["train4"]
                    y_train = Y_train["train4"]
                    x_val = X_train["val4"]
                    y_val = Y_train["val4"]
                    x_test = X_train["test4"]
                    y_test = Y_train["test4"]
                elif (self.central_epoch >= 5 * max_epochs) and (self.central_epoch < 6 * max_epochs):
                    x_train = X_train["train5"]
                    y_train = Y_train["train5"]
                    x_val = X_train["val5"]
                    y_val = Y_train["val5"]
                    x_test = X_train["test5"]
                    y_test = Y_train["test5"]
                elif (self.central_epoch >= 6 * max_epochs) and (self.central_epoch < 7 * max_epochs):
                    x_train = X_train["train6"]
                    y_train = Y_train["train6"]
                    x_val = X_train["val6"]
                    y_val = Y_train["val6"]
                    x_test = X_train["test6"]
                    y_test = Y_train["test6"]
                elif (self.central_epoch >= 7 * max_epochs) and (self.central_epoch < 8 * max_epochs):
                    x_train = X_train["train7"]
                    y_train = Y_train["train7"]
                    x_val = X_train["val7"]
                    y_val = Y_train["val7"]
                    x_test = X_train["test7"]
                    y_test = Y_train["test7"]
                elif (self.central_epoch >= 8 * max_epochs) and (self.central_epoch < 9 * max_epochs):
                    x_train = X_train["train8"]
                    y_train = Y_train["train8"]
                    x_val = X_train["val8"]
                    y_val = Y_train["val8"]
                    x_test = X_train["test8"]
                    y_test = Y_train["test8"]
                elif self.central_epoch >= 9 * max_epochs:
                    x_train = X_train["train9"]
                    y_train = Y_train["train9"]
                    x_val = X_train["val9"]
                    y_val = Y_train["val9"]
                    x_test = X_train["test9"]
                    y_test = Y_train["test9"]

                # xtrain = pd.DataFrame(x_train)
                # print(y_train)
                # ytrain = pd.concat((pd.DataFrame(y_train[0]), pd.DataFrame(y_train[1])), axis=1)#pd.DataFrame(y_train)

                # get_target = lambda df: (df.iloc[:, 0].values.astype('int32'), df.iloc[:, 1].values.astype('int32'))

                # x_train, x_val, y_train, y_val = train_test_split(xtrain, ytrain, test_size=0.2)
                # x_train = x_train.reset_index(drop=True)
                # y_train = y_train.reset_index(drop=True)
                # y_train = get_target(y_train)

                # x_val = x_val.reset_index(drop=True)
                # y_val = y_val.reset_index(drop=True)
                # y_val = get_target(y_val)

                # standardize_cols = ['age','wbc_cont','hb_cont','plt','bm_blast']
                # standardize_cols = ['age','i_cpss_score','i_cpss_risk']

                ##standardize_cols = ['AOD','Neutrophils','Hemoglobin','Platelets','BMB']
                ##features = list(x_train.columns)
                ##standardize_index = []
                ##for i in standardize_cols:
                ##    if i in features:
                ##        standardize_index.append(x_train.columns.get_loc(i))
                ##scaler = MinMaxScaler()
                ##x_train.iloc[:, standardize_index] = scaler.fit_transform(x_train.iloc[:, standardize_index])
                ##scaler = MinMaxScaler()
                ##x_val.iloc[:, standardize_index] = scaler.fit_transform(x_val.iloc[:, standardize_index])
                ##print('standardize')
                ##print(x_train.shape)

                trainloader = (np.array(x_train), y_train)
                valloader = (np.array(x_val), y_val)
                testloader = (np.array(x_test), y_test)

                #print('CV PHASE OF DATA PREPARATION')
                #print('CENTRAL EPOCH ', self.central_epoch)
                ##print(x_train.columns)
                #print(x_val.columns)
                #print(x_test.columns)

                for k in np.arange(0, num_epochs_int, 1):
                    #print('TRAIN DATA BEFORE FIT')
                    #print(trainloader[0].shape)
                    #print(trainloader[0][0, :])
                    #print(trainloader[0], trainloader[1])

                    #print('VAL DATA BEFORE FIT')
                    #print(valloader[0].shape)
                    #print(valloader[0][0, :])
                    #print(valloader[0], valloader[1])

                    #print('TEST DATA BEFORE FIT')
                    #print(testloader[0].shape)
                    #print(testloader[0][0, :])
                    #print(testloader[0], testloader[1])

                    # print('BEFORE LR_FINDER')
                    #print('weights: ', [val.cpu().numpy() for _, val in self.model.net.state_dict().items()])

                    # lrfinder = self.model.lr_finder(trainloader[0], trainloader[1], batch_size,
                    #                                  tolerance=10)
                    # lr = lrfinder.get_best_lr()
                    # self.model.optimizer.set_lr(0.01)

                    #print('CENTRAL EPOCH')
                    #print(self.central_epoch)

                    # print('LEARNING RATE')
                    # print(lr)

                    ##print('BEFORE FIT')
                    ##print('weights: ', [val.cpu().numpy() for _, val in self.model.net.state_dict().items()])
                    #print('model parameters: ', self.model.optimizer.state_dict())

                    self.model.fit(trainloader[0], trainloader[1], batch_size, epochs=1,  # callbacks, verbose,
                                   val_data=valloader, val_batch_size=batch_size)
                    # print('fitting roud ', k , ': ', self.get_weight())

                    #print('AFTER FIT')
                    #print('weights: ', [val.cpu().numpy() for _, val in self.model.net.state_dict().items()])
                    #print('model parameters: ', self.model.optimizer.state_dict())
                    #print(testloader[0])
                    _ = self.model.compute_baseline_hazards(testloader[0], testloader[1])
                    surv = self.model.predict_surv_df(testloader[0])
                    c_index = obtain_c_index(surv, testloader[1][0], testloader[1][1])

                    #print('AFTER C_INDEX CALCULATION')
                    #print('weights: ', [val.cpu().numpy() for _, val in self.model.net.state_dict().items()])

                    mlflow.log_metric(f"c_index", f'{c_index:.3f}')
                    FileSave(self.metrics_file,str(str(self.central_epoch)+','+str(float(c_index))+'\n'))
                    # mlflow.log_metric(f"ibs", f'{ibs:.3f}')
                    self.central_epoch = self.central_epoch + 1
                    #if self.central_epoch in [99,199,299,399,499,599,699,799,899,999]:
                    print('C-INDEX AT THE END OF FIT:')
                    print(c_index)

                    if self.central_epoch == 10*max_epochs:
                        FileSave(self.metrics_file,str('Number of patients for train:'+str(self.patients_train)+'\n'))
                        FileSave(self.metrics_file,str('Number of patients for test:'+str(self.patients_test)+'\n'))

            #def get_weight(self):
            #    print('GET PARAMETERS...')
            #    # print('weights: ', [val.cpu().numpy() for _, val in self.model.net.state_dict().items()])
            #    return [val.cpu().numpy() for _, val in self.model.net.state_dict().items()]

            def set_weight(self, parameters):
                print('SET PARAMETERS...')
                params_dict = zip(self.model.net.state_dict().keys(), parameters)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                state_dict['net.0.batch_norm.num_batches_tracked'] = torch.Tensor(1)  # np.array(1)
                state_dict['net.1.batch_norm.num_batches_tracked'] = torch.Tensor(1)  # np.array(1)
                state_dict['net.2.batch_norm.num_batches_tracked'] = torch.Tensor(1)  # np.array(1)
                return self.model.net.load_state_dict(state_dict, strict=False)  # True

            def get_weight(self):# -> List[torch.Tensor]:
                # Estrai tutti i parametri come tensori CPU clonati
                return [val.detach().clone().cpu().numpy() for _, val in self.model.net.state_dict().items()]

            # def set_parameters(model, parameters: List[np.ndarray]):
            #    print('SET PARAMETERS...')
            #    params_dict = zip(model.net.state_dict().keys(), parameters)
            #    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            #    state_dict['net.0.batch_norm.num_batches_tracked'] = torch.Tensor(1)
            #    state_dict['net.1.batch_norm.num_batches_tracked'] = torch.Tensor(1)
            #    state_dict['net.2.batch_norm.num_batches_tracked'] = torch.Tensor(1)
            #    # print(state_dict)
            #    model.net.load_state_dict(state_dict, strict=False)  # True)

            #def set_weight(self, parameters):
            #    print('SET PARAMETERS...')
            #    state_dict = self.model.net.state_dict()
            #    new_state_dict = OrderedDict()#

            #    keys = list(state_dict.keys())
            #    if len(keys) != len(parameters):
            #        raise ValueError(f"Mismatch: {len(keys)} keys vs {len(parameters)} parameters")

            #    for k, v in zip(keys, parameters):
            #        if v.shape != state_dict[k].shape:
            #            print(f"[WARNING] Shape mismatch for {k}: {v.shape} vs {state_dict[k].shape}")
            #            new_state_dict[k] = state_dict[k]  # fallback: originale
            #        else:
            #            new_state_dict[k] = v

            #    return self.model.net.load_state_dict(new_state_dict, strict=False)

            def evaluate(self, X_test, Y_test) -> Tuple[float, float]:
                print('EVALUATE!')
                print("CENTRAL EPOCH IN EVALUATE: ", self.central_epoch)

                max_epochs = params['max_epochs']

                if self.central_epoch <= max_epochs:
                    x_test = X_test["test0"]
                    y_test = Y_test["test0"]
                elif (self.central_epoch > max_epochs) and (self.central_epoch <= 2 * max_epochs):
                    x_test = X_test["test1"]
                    y_test = Y_test["test1"]
                elif (self.central_epoch > 2 * max_epochs) and (self.central_epoch <= 3 * max_epochs):
                    x_test = X_test["test2"]
                    y_test = Y_test["test2"]
                elif (self.central_epoch > 3 * max_epochs) and (self.central_epoch <= 4 * max_epochs):
                    x_test = X_test["test3"]
                    y_test = Y_test["test3"]
                elif (self.central_epoch > 4 * max_epochs) and (self.central_epoch <= 5 * max_epochs):
                    x_test = X_test["test4"]
                    y_test = Y_test["test4"]
                elif (self.central_epoch > 5 * max_epochs) and (self.central_epoch <= 6 * max_epochs):
                    x_test = X_test["test5"]
                    y_test = Y_test["test5"]
                elif (self.central_epoch > 6 * max_epochs) and (self.central_epoch <= 7 * max_epochs):
                    x_test = X_test["test6"]
                    y_test = Y_test["test6"]
                elif (self.central_epoch > 7 * max_epochs) and (self.central_epoch <= 8 * max_epochs):
                    x_test = X_test["test7"]
                    y_test = Y_test["test7"]
                elif (self.central_epoch > 8 * max_epochs) and (self.central_epoch <= 9 * max_epochs):
                    x_test = X_test["test8"]
                    y_test = Y_test["test8"]
                elif self.central_epoch > 9 * max_epochs:
                    x_test = X_test["test9"]
                    y_test = Y_test["test9"]

                get_target = lambda df: (df.iloc[:, 0].values.astype('int32'), df.iloc[:, 1].values.astype('int32'))

                test = x_test  # .astype('float32')
                ytest = pd.concat((pd.DataFrame(y_test[0]), pd.DataFrame(y_test[1])), axis=1)
                test_surv = get_target(ytest)
                test_loader_eval = (np.array(test), test_surv)

                print('COMPUTING C INDEX IN EVALUATE')

                # ev = EvalSurv(surv, test_loader_eval[1][0], test_loader_eval[1][1], censor_surv='km')
                # c_index = ev.concordance_td()

                # _ = self.model.compute_baseline_hazards(test_loader_eval[0], test_loader_eval[1])
                #print(test_loader_eval[0])
                #print('weights: ', [val.cpu().numpy() for _, val in self.model.net.state_dict().items()])

                _ = self.model.compute_baseline_hazards(test_loader_eval[0], test_loader_eval[1])
                surv = self.model.predict_surv_df(test_loader_eval[0])
                c_index = obtain_c_index(surv, test_loader_eval[1][0], test_loader_eval[1][1])

                loss = 0.0
                print("DEBUG c_index =", c_index, "| Type:", type(c_index))
                print('C-INDEX: ', c_index)
                return float(loss), c_index #{"c_index": float(c_index)}

            def load_data(self, path):
                return load_data(path)

        return PythonModelWrapper(dataset_path=dataset_path)


# Model instantiation and registration
import cloudpickle

model = FLModel()
# with open(r'C:\Users\gasti\Documents\Progetti\Genomed4All\Macchine_GCP\Models\DeepSurv_v3_0_5\DeepSurv_v1.pk', 'wb') as f:
with open(r'/mnt/c/users/lenovo/desktop/FedSDS_biased_flower/DeepSurv_cv_lgg.pk', 'wb') as f: #_scaled75
    cloudpickle.dump(model, f)

# import mlflow as mlf
# mlflow.set_tracking_uri("https://urldefense.proofpoint.com/v2/url?u=http-3A__localhost-3A5002_&d=DwIGAg&c=5rLNXN0mp_7LMh3Fds96xpjyD06ZuE2RU7zikolS0lg&r=InnoW2J7WO2GyBp1EYkPTuFH7X5gCTIwBjlI02urYgc&m=txI-RX7PkACrYpay7SwX5NXUz0PjXpxO0i1t9mV9tBsiw6ZBmJaRCLwAFTCnWewe&s=isz5ZMUSgIZGWIKTxt1qE-Bccaz03YnZKxqh3tZKgiQ&e= ")
##mlflow.set_tracking_uri("https://urldefense.proofpoint.com/v2/url?u=https-3A__playground.gpapdev.cnag.eu_mlflow_&d=DwIGAg&c=5rLNXN0mp_7LMh3Fds96xpjyD06ZuE2RU7zikolS0lg&r=InnoW2J7WO2GyBp1EYkPTuFH7X5gCTIwBjlI02urYgc&m=txI-RX7PkACrYpay7SwX5NXUz0PjXpxO0i1t9mV9tBsiw6ZBmJaRCLwAFTCnWewe&s=zjpDz291yhhI1Sony4sm0GoZOL3cGuFBDyaqjvbFnWU&e= ")
# mlflow.set_experiment("test2")

# mlf.pyfunc.log_model(
#         python_model=model,
#         artifact_path="modello_pytorch_new",
#         registered_model_name="DeepSurv_data",
#     )
