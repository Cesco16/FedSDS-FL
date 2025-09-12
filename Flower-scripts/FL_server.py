
# Author: Francesco Casadei and Luciana Carota
# Email: francesco.casadei20@unibo.it, luciana.carota@unibo.it
# Date: 19/05/2025

# Import libraries
import pickle
import flwr as fl
import sys

connection_port = '8005'
ip = '000.00.00.00'

file_name = './DeepSurv_cv_mds.pk'

with open(file_name, "rb") as f:
    model = pickle.load(f).create_strategy()

# Start Flower server
fl.server.start_server(
    server_address=str(ip+':'+ connection_port),
    config=fl.server.ServerConfig(num_rounds=100),
    strategy=model
)
