# FedSDS-FL

Repository of the scripts used for the analyses done in the research study: "Leveraging biased FedSDS as a Federated Learning strategy for research on non-IID multi-disorder studies".

## Flower-scripts

Scripts used to deploy the FL environment with flower and to run the different training of models.

1. FL_server.py: deploy the FL central server to aggregate weights
2. FL_nodeN.py: deploy and run a federated client, connected to the central server
3. main_model_DATA.py: create the federated model and load and preprocess the chosen dataset

## scripts

1. select_seed.py: select the best synthetic dataset using a Random Forest classifier
2. bias_data.py: bias the selected synthetic dataset in each local node
3. plot.py: generate plots

## models

It contains the federated models generate using the main_.py scripts. Models are stored in pickle (.pk) format.
