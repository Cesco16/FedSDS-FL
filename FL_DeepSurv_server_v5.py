
# Author: Francesco Casadei
# Email: francesco.casadei20@unibo.it
# Date: 19/02/2024

# Import libraries
import pickle
import flwr as fl
import sys

connection_port = 8005 #8437

print(connection_port)


##file_name = './DeepSurv_cv_val_MDS_platform_1nodes_v2.pk'
##file_name = './DeepSurv_cv_uniquetest_MDS_platform_1nodes.pk'
##file_name = './DeepSurv_cv_val_MDS_platform_1nodes_v2.pk'
file_name = './DeepSurv_cv_mds.pk'
##file_name = './DS_uniquetest_MDS_platform_v1_noinit.pk'

with open(file_name, "rb") as f:
    model = pickle.load(f).create_strategy()

# Start Flower server
fl.server.start_server(
#    server_address="10.255.219.120:8082",
    server_address=str('172.17.34.44:'+str(connection_port)),
    #server_address=str('137.204.51.140:'+str(connection_port)),
    config=fl.server.ServerConfig(num_rounds=100),
    strategy=model
)
