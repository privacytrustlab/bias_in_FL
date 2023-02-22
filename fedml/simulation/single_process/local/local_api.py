import copy
import logging
import random

import numpy as np
import torch
import wandb
import pickle
import os
from .client import Client
import logging
import copy

class LocalAPI(object):
    def __init__(self, args, device, dataset, model,model_trainer=None):
        self.device = device
        self.args = args
        [
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            val_data_local_dict,
            class_num,
        ] = dataset
        
        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.val_data_local_dict = val_data_local_dict
        self.local_model_dict = {}


        logging.info("model = {}".format(model))
        # self.model_trainer = model_trainer
        # logging.info("self.model_trainer = {}".format(self.model_trainer))

        self._setup_clients(
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            val_data_local_dict,model_trainer,
        )

    def _setup_clients(
        self,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        val_data_local_dict,
        model_trainer,
    ):
        logging.info("############setup_clients (START)#############")
        for client_idx in self.args.users:
            c = Client(
                client_idx,
                train_data_local_dict[client_idx],
                test_data_local_dict[client_idx],
                val_data_local_dict[client_idx],
                train_data_local_num_dict[client_idx],
                self.args,
                self.device,
                copy.deepcopy(model_trainer),
            )
            self.client_list.append(c)
            self.local_model_dict[client_idx] = c.get_local_weight()
        logging.info("############setup_clients (END)#############")

    def train(self):
       
        for round_idx in range(self.args.comm_round):

            logging.info("################Local epoch : {}".format(round_idx))
            logging.info(self.args.users)
            for idx, client in zip(self.args.users, self.client_list):
    
                self.local_model_dict[idx] = copy.deepcopy(client.train(copy.deepcopy(self.local_model_dict[idx])))

          
            if round_idx % self.args.save_epoches == 0: 
                    with open(os.path.join(self.args.run_folder, "%s_at_%s.pt" %(self.args.save_model_name,round_idx)), 'wb') as f:
                        pickle.dump(self.local_model_dict, f)

            if round_idx == self.args.comm_round - 1 or round_idx % self.args.frequency_of_the_test == 0:
                self._local_test_on_all_clients(round_idx)

    def _client_sampling(self):
       return False

    def _generate_validation_set(self, num_samples=10000):
        return False

    def _aggregate(self, w_locals):
        return False

    def _aggregate_noniid_avg(self, w_locals):
        return False

    def _local_test_on_all_clients(self, round_idx):

        logging.info("################local_test_on_all_clients : {}".format(round_idx))
        
        train_metrics = {"num_samples": [], "num_correct": [], "losses": [], "eo_gap":[],"dp_gap":[]}

        test_metrics = {"num_samples": [], "num_correct": [], "losses": [], "eo_gap":[],"dp_gap":[]}


        for idx, client in zip(self.args.users,self.client_list):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """

            # train data
            train_local_metrics = client.local_test(False)
            train_metrics["num_samples"].append(copy.deepcopy(train_local_metrics["test_total"]))
            train_metrics["num_correct"].append(copy.deepcopy(train_local_metrics["test_correct"]))
            train_metrics["losses"].append(copy.deepcopy(train_local_metrics["test_loss"]))
            train_metrics["eo_gap"].append(copy.deepcopy(train_local_metrics["eo_gap"]))
            train_metrics["dp_gap"].append(copy.deepcopy(train_local_metrics["dp_gap"]))


            # test data
            test_local_metrics = client.local_test(True)
            test_metrics["num_samples"].append(copy.deepcopy(test_local_metrics["test_total"]))
            test_metrics["num_correct"].append(copy.deepcopy(test_local_metrics["test_correct"]))
            test_metrics["losses"].append(copy.deepcopy(test_local_metrics["test_loss"]))
            test_metrics["eo_gap"].append(copy.deepcopy(test_local_metrics["eo_gap"]))
            test_metrics["dp_gap"].append(copy.deepcopy(test_local_metrics["dp_gap"]))

          

          
                


        train_acc = sum(train_metrics["num_correct"]) / sum(train_metrics["num_samples"])
        train_loss = sum(train_metrics["losses"]) / sum(train_metrics["num_samples"])
        train_dp_gap = sum(train_metrics["dp_gap"])/len(self.args.users)
        train_eo_gap = sum(train_metrics["eo_gap"])/len(self.args.users)

        test_acc = sum(test_metrics["num_correct"]) / sum(test_metrics["num_samples"])
        test_loss = sum(test_metrics["losses"]) / sum(test_metrics["num_samples"])
        test_dp_gap = sum(test_metrics["dp_gap"])/len(self.args.users)
        test_eo_gap = sum(test_metrics["eo_gap"])/len(self.args.users)
        
        logging.info('Train acc: {} Train Loss: {}, Test acc: {} Test Loss: {}'.format(train_acc,train_loss, test_acc,test_loss))
        logging.info('Train dp gap: {} Train eo gap: {}, Test dp gap: {} Test eo gap: {}'.format(train_dp_gap,train_eo_gap, test_dp_gap,test_eo_gap))


    def _local_test_on_validation_set(self, round_idx):
        return False

    def save(self):
        with open(os.path.join(self.args.run_folder, "%s.pt" %(self.args.save_model_name)), 'wb') as f:
            pickle.dump(self.local_model_dict, f)