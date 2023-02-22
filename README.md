# Bias Propagation in Federated Learning

This is the code repository for the paper titled [Bias Propagation in Federated Learning](https://openreview.net/forum?id=V7CYzdruWdm) which was accepted to the International Conference on Learning Representations (ICLR) 2023.

If you have any questions, feel free to email Hongyan (hongyan@comp.nus.edu.sg).

## Introduction

In our paper, we show that participating in federated learning can be detrimental to group fairness. In fact, the bias of a few biased parties against under-represented groups (identified by sensitive attributes such as gender or race) propagates through the network to all parties. On naturally partitioned real-world datasets, we analyze and explain bias propagation in federated learning. Our analysis reveals that biased parties unintentionally yet stealthily encode their bias in a small number of model parameters, and throughout the training, they steadily increase the dependence of the global model on sensitive attributes. What is important to highlight is that the experienced bias in federated learning is higher than what parties would otherwise encounter in centralized training with a model trained on the union of all their data. This indicates that the bias is due to the algorithm. Our work calls for auditing group fairness in federated learning, and designing learning algorithms that are robust to bias propagation.

## Dependencies

Our implementation of Federated Learning is based on the [FedML library](https://github.com/FedML-AI/FedML) and we use the ML tasks provided by forlfork table.

and we tested our code based on the tested on `Python 3.8.13` and `cuda 11.4`. The essential environments are listed in the `environment.yml` file. Run the following command to create the conda environment:

```
conda env create -f environment.yml
```

### Usage

#### 1. Training the models for different settings.

To run federated learning on Income dataset, use the command:

```
python main.py --cf config/config_fedavg_income.yaml
```

Similarly, to run the centralized training, using the following command:

```
python main.py --cf config/config_centralized_income.yaml
```

Standalone training:

```
python main.py --cf config/config_standalone_income.yaml
```

We report the average results over five different runs. Thus, to reproduce the results, run the command five times with different random seeds, which is indicated by common_args.random_seed in the YAML file.

To get the results on other datasets (e.g., Health, employment dataset), run the `main.py` file with `config/config_standalone_{dataset}.yaml`, where the dataset can be health, employment, or income.

#### 2.

###

###

In order to get the results and plots presented in the paper, please follow the following

The implementation of the FedAvg is in `fedavg` folder, and the implementation of other FL algorithms is in `other_fl`. Next, we give examples of how to run FedAvg algorithm.
