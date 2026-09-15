# Author: Francesco Casadei, Luciana Carota
# Email: francesco.casadei20@unibo.it, luciana.carota@unibo.it
# Date: 19/05/2025
# Modified by Luciana Carota for a unique test file and 5 nodes 22/11/2024

# MODEL DEEPSURV FROM PYCOX

# Import libraries
import mlflow
from collections import OrderedDict
from typing import Tuple, Dict
import os
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURAZIONE DELLA STRATEGY DI FEDERATED LEARNING
# Cambia questi due valori per passare da FedAvg a FedProx o FedNova.
# Puoi anche sovrascrivere STRATEGY_NAME da riga di comando lanciando:
#   python FL_server.py fedprox     (oppure: fedavg | fednova)
# ============================================================================
STRATEGY_NAME = "fednova"   # "fedavg" | "fedprox" | "fednova"
PROXIMAL_MU = 0.7#0.1           # usato solo se STRATEGY_NAME == "fedprox"


def load_data(data_path_train) -> Tuple[Dict, Dict]:
    import pandas as pd
    from os.path import realpath
    from os.path import dirname
    from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder

    # all the features
    continuous = ['age_at_index', 'age_at_diagnosis']
    binary = ['gender', 'prior_malignancy', 'prior_treatment']
    categorical = ['race', 'morphology', 'primary_diagnosis', 'year_of_diagnosis', 'tissue_or_organ_of_origin','treatment_or_therapy',
                   'ajcc_pathologic_m', 'ajcc_pathologic_n', 'ajcc_pathologic_stage', 'ajcc_pathologic_t','ajcc_staging_system_edition']
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
    r1 = pd.read_csv(data_path_train + 'brca_.csv')
    # TEST DATA
    t1 = pd.read_csv(data_path_train + 'testbrca_.csv')

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

    mode = 'iso'

    if mode == 'fed_avg':
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

    # encode the categorical variables in the training set
    categories = [
        [0, 1, 2, 3],  # race
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],  # morphology
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],  # primary_diagnosis
        [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],  # year_of_diagnosis
        [0, 1, 2, 3],  # tissue_or_organ_of_origin
        [0, 1, 2],  # treatment_or_therapy
        [0, 1, 2, 3],  # ajcc_pathologic_m
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],  # ajcc_pathologic_n
        [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # ajcc_pathologic_stage
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # ajcc_pathologic_t
        [0, 1, 2, 3, 4],  # ajcc_staging_system_edition
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
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],  # morphology
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],  # primary_diagnosis
        [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],  # year_of_diagnosis
        [0, 1, 2, 3],  # tissue_or_organ_of_origin
        [0, 1, 2],  # treatment_or_therapy
        [0, 1, 2, 3],  # ajcc_pathologic_m
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],  # ajcc_pathologic_n
        [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # ajcc_pathologic_stage
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # ajcc_pathologic_t
        [0, 1, 2, 3, 4],  # ajcc_staging_system_edition
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
    def create_strategy(self, num_rounds=100, strategy_name=None, proximal_mu=None):
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
        from flwr.common import (
            FitRes, Scalar, Parameters, parameters_to_ndarrays, ndarrays_to_parameters,
        )

        # scegli la strategy: parametro esplicito > costante globale in cima al file
        strategy_name = (strategy_name if strategy_name is not None else STRATEGY_NAME).lower()
        proximal_mu = PROXIMAL_MU if proximal_mu is None else proximal_mu

        class AggregateCustomMetricStrategy(flwr.server.strategy.FedAvg):
            """Strategy originale (FedAvg), lasciata invariata per compatibilita'."""
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

                return aggregated_parameters, aggregated_metrics

        class FedNovaStrategy(flwr.server.strategy.FedAvg):
            """
            FedNova (Wang et al., 2020 - https://arxiv.org/abs/2007.07481).

            Flower NON include FedNova come strategy nativa (a differenza di
            FedProx), quindi la implemento qui come sottoclasse di FedAvg che
            cambia solo l'aggregazione dei pesi (aggregate_fit).

            Ogni client esegue tau_i step locali (che qui possono variare da
            nodo a nodo, ad es. per dati/epoche diverse) e riporta questo
            valore nei metrics di fit() sotto la chiave "tau" (vedi
            FL_node1.py / model.get_tau()). Il server normalizza il
            contributo di ciascun client per il proprio tau_i e poi lo scala
            per tau_eff = media pesata dei tau_i, correggendo il bias verso i
            client che fanno piu' step locali (il problema di "objective
            inconsistency" descritto nel paper).

                new_weights = old_weights - tau_eff * sum_i [ p_i * (delta_i / tau_i) ]
            """

            def aggregate_fit(self, server_round, results, failures):
                if not results:
                    return None, {}
                if failures and not self.accept_failures:
                    return None, {}

                if getattr(self, "current_parameters", None) is None:
                    # nessuno storico dei parametri globali: fallback su FedAvg puro
                    return super().aggregate_fit(server_round, results, failures)

                global_ndarrays = parameters_to_ndarrays(self.current_parameters)
                total_examples = sum(fit_res.num_examples for _, fit_res in results)

                tau_eff = 0.0
                weighted_normalized_deltas = None
                for _, fit_res in results:
                    p_i = fit_res.num_examples / total_examples
                    tau_i = max(float(fit_res.metrics.get("tau", 1)), 1.0)
                    tau_eff += p_i * tau_i

                    client_ndarrays = parameters_to_ndarrays(fit_res.parameters)
                    delta_i = [g - l for g, l in zip(global_ndarrays, client_ndarrays)]
                    normalized_delta_i = [d / tau_i for d in delta_i]

                    if weighted_normalized_deltas is None:
                        weighted_normalized_deltas = [p_i * d for d in normalized_delta_i]
                    else:
                        weighted_normalized_deltas = [
                            acc + p_i * d for acc, d in zip(weighted_normalized_deltas, normalized_delta_i)
                        ]

                new_ndarrays = [
                    g - tau_eff * d for g, d in zip(global_ndarrays, weighted_normalized_deltas)
                ]
                self.current_parameters = ndarrays_to_parameters(new_ndarrays)

                metrics_aggregated = {}
                if self.fit_metrics_aggregation_fn:
                    fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
                    metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

                return self.current_parameters, metrics_aggregated

            def initialize_parameters(self, client_manager):
                params = super().initialize_parameters(client_manager)
                self.current_parameters = params
                return params

            def configure_fit(self, server_round, parameters, client_manager):
                self.current_parameters = parameters
                return super().configure_fit(server_round, parameters, client_manager)

        common_kwargs = dict(
            min_available_clients=2,
            min_evaluate_clients=2,
            min_fit_clients=2,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
        )

        if strategy_name == "fedprox":
            # --- Strategy NATIVA di Flower, nessuna implementazione custom ---
            print(f"[create_strategy] Uso FedProx nativo di Flower (proximal_mu={proximal_mu})")
            return flwr.server.strategy.FedProx(proximal_mu=proximal_mu, **common_kwargs)
        elif strategy_name == "fednova":
            print("[create_strategy] Uso FedNova (strategy custom, vedi FedNovaStrategy)")
            return FedNovaStrategy(**common_kwargs)
        elif strategy_name == "fedavg":
            print("[create_strategy] Uso FedAvg (strategy originale, invariata)")
            return AggregateCustomMetricStrategy(**common_kwargs)
        else:
            raise ValueError(
                f"strategy_name '{strategy_name}' non riconosciuto. "
                "Usa 'fedavg', 'fedprox' oppure 'fednova'."
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
            'in_features': 0,  # dinaically created in PhytonWrapper
            'num_nodes': [32, 32, 32],  # [32, 32],
            'out_features': 1,
            'batch_norm': True,  # False,
            'dropout': 0.5,  # 0.7,
            'output_bias': False,
            'activation': torch.nn.SELU,  # torch.nn.ReLU,
            'w_decay': 0,
            'batch_size': 512,  # 126
            'num_epochs_int': 10,
            'max_epochs': 100,
            'lr': 0.01,  # 0.01
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

        class ProxCoxPHLoss(torch.nn.Module):
            """
            Wrapper attorno alla loss originale di CoxPH (CoxPHLoss di pycox)
            che aggiunge il termine prossimale di FedProx (Li et al., 2018):

                loss_locale = loss_originale + (mu/2) * ||w_locale - w_globale||^2

            Se mu == 0 (default, e sempre il caso per FedAvg/FedNova) si
            comporta esattamente come la loss originale: nessun impatto sulle
            altre strategy. torchtuples chiama `self.loss(*out, *target)` ad
            ogni batch (vedi torchtuples.Model.compute_metrics), quindi basta
            sostituire l'attributo `model.loss` con questo wrapper.
            """

            def __init__(self, net, base_loss):
                super().__init__()
                self.net = net
                self.base_loss = base_loss
                self.mu = 0.0
                self.global_params = None

            def set_mu(self, mu: float):
                self.mu = float(mu)

            def snapshot_global_params(self):
                # da chiamare subito dopo aver caricato i pesi globali nel
                # modello (cioe' dopo set_weight, prima del training locale)
                self.global_params = [p.detach().clone() for p in self.net.parameters()]

            def forward(self, *args):
                loss = self.base_loss(*args)
                if self.mu > 0 and self.global_params is not None:
                    prox_term = sum(
                        torch.sum((p - g) ** 2)
                        for p, g in zip(self.net.parameters(), self.global_params)
                    )
                    loss = loss + (self.mu / 2) * prox_term
                return loss

        class PythonModelWrapper:
            def __init__(self, dataset_path):
                self.data = self.load_data(dataset_path)

                in_features = len(self.data[0][0]['train0'].columns)
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
                data_path_cindex = data_path + "//cindex_review_unbiased.csv"

                self.model = CoxPH(net, tt.optim.Adam(weight_decay=params['w_decay']))
                self.model.net.to(device)
                # Avvolgo la loss originale per poter aggiungere il termine
                # prossimale di FedProx quando serve (mu=0 => nessun effetto).
                self.model.loss = ProxCoxPHLoss(self.model.net, self.model.loss)

                self.cindex = []
                self.training_stats = {}
                self.central_epoch = 0
                self.metrics_file = data_path_cindex
                self.last_tau = 0  # numero di step locali dell'ultima fit() (usato da FedNova)

            def predict(self, model_input):
                return self.model.predict(model_input)

            def fit(self, X_train, Y_train, epochs=100, batch_size=params['batch_size'], steps_per_epoch=1, config=None):

                from sklearn.preprocessing import MinMaxScaler, StandardScaler
                from sklearn.model_selection import train_test_split

                num_epochs_int = params['num_epochs_int']
                batch_size = params['batch_size']
                max_epochs = params['max_epochs']

                print('\n Try to fit ...')
                training_stats = {'c_index': [], 'ibs': [], 'epochs': [], 'num_round': [],
                                  'node_name': []}  # To store training data

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
                    # Riavvolgo la loss anche qui: il modello e' stato ricreato da zero.
                    self.model.loss = ProxCoxPHLoss(self.model.net, self.model.loss)


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


                trainloader = (np.array(x_train), y_train)
                valloader = (np.array(x_val), y_val)
                testloader = (np.array(x_test), y_test)

                # --- FedProx: legge proximal_mu dal config inviato dal server -----
                # Il server FedProx nativo di Flower inietta automaticamente la
                # chiave "proximal_mu" nel config passato a fit() dal client.
                # Se la strategy e' FedAvg o FedNova, config non conterra' questa
                # chiave (o sara' None) e il termine prossimale resta disattivato.
                proximal_mu = 0.0
                if config is not None:
                    proximal_mu = config.get('proximal_mu', 0.0)
                if isinstance(self.model.loss, ProxCoxPHLoss):
                    self.model.loss.set_mu(proximal_mu)
                    # I pesi correnti del net sono quelli globali appena ricevuti
                    # (set_weight viene chiamato dal client PRIMA di fit()), quindi
                    # questo e' il momento giusto per fare lo snapshot.
                    self.model.loss.snapshot_global_params()

                # --- FedNova: conta gli step locali effettivi (tau_i) -------------
                # ogni chiamata a self.model.fit(..., epochs=1, ...) qui sotto fa
                # ceil(n_train/batch_size) step di ottimizzazione; il ciclo la
                # ripete num_epochs_int volte.
                steps_per_epoch_est = int(np.ceil(len(trainloader[0]) / batch_size))
                self.last_tau = steps_per_epoch_est * num_epochs_int

                for k in np.arange(0, num_epochs_int, 1):

                    self.model.fit(trainloader[0], trainloader[1], batch_size, epochs=1,  # callbacks, verbose,
                                   val_data=valloader, val_batch_size=batch_size)
                    _ = self.model.compute_baseline_hazards(testloader[0], testloader[1])
                    surv = self.model.predict_surv_df(testloader[0])
                    c_index = obtain_c_index(surv, testloader[1][0], testloader[1][1])

                    mlflow.log_metric(f"c_index", f'{c_index:.3f}')
                    FileSave(self.metrics_file,str(str(self.central_epoch)+','+str(float(c_index))+'\n'))
                    self.central_epoch = self.central_epoch + 1
                    print('C-INDEX AT THE END OF FIT:')
                    print(c_index)

                    if self.central_epoch == 10*max_epochs:
                        FileSave(self.metrics_file,str('Number of patients for train:'+str(self.patients_train)+'\n'))
                        FileSave(self.metrics_file,str('Number of patients for test:'+str(self.patients_test)+'\n'))


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

            def get_tau(self) -> int:
                """Numero di step locali (batch) eseguiti nell'ultima fit(). Usato da FedNova."""
                return int(self.last_tau)

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
#with open(r'/mnt/c/users/lenovo/desktop/FedSDS_biased_flower/DeepSurv_cv_mds.pk', 'wb') as f:
#    cloudpickle.dump(model, f)
with open(r'/mnt/c/users/lenovo/desktop/FedSDS_review/DeepSurv_cv_brca_rev2.pk', 'wb') as f:
    cloudpickle.dump(model, f)
