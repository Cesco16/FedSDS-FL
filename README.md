# FedSDS-FL

Repository of the scripts used for the analyses done in the research study: "Leveraging biased FedSDS as a Federated Learning strategy for research on non-IID multi-disorder studies".

## Flower-scripts

Scripts used to deploy the FL environment with flower and to run the different training of models.

1. FL_server.py: deploy the FL central server to aggregate weights
2. FL_nodeN.py: deploy and run a federated client, connected to the central server
3. main_ds_DATA.py: deploy the FL strategy (Isolated training, FedAvg, FedProx, FedNova, FedSDS), create the federated model and load and preprocess the chosen dataset (MDS, LGG, BRCA)

## scripts

0. main_preprocess_sa_data.py: preprocess survival analysis data (MDS, LGG, BRCA) before synthetic data generation with VAE-BGM
1. main_sa_data.py: synthetic data generation through VAE-BGM
2. select_seed.py: select the best synthetic dataset using a Random Forest classifier
3. bias_data.py: bias the selected synthetic dataset in each local node
4. plot.py: generate plots
5. compute_privacy.py: compute privacy metrics and generate pdf and cdf histograms

## models

It contains the federated models generate using the main_.py scripts. Models are stored in pickle (.pk) format.

## venvs

It contains the packages needed to create the Python virtual environments:
- vaebgm: environment to run synthetic data generation
- genomed4all: environment to run FL training
