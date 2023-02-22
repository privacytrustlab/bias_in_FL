import pickle

import altair as alt
import numpy as np
import pandas as pd
import torch

alt.renderers.enable("mimetype")
import argparse
import os

""""Get the prediction for the trained models"""

census_input_shape_dict = {
    "income": 54,
    "health": 154,
    "employment": 109,
    "travel": 16,
    "resident": 21,
}


hidden_shape_dict = {"income": 32, "health": 64, "employment": 64}


class TwoNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_outdim, output_dim):
        super(TwoNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_outdim)
        self.relu = torch.nn.ReLU(inplace=True)
        self.fc2 = torch.nn.Linear(hidden_outdim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def get_prediction(model, x, y):
    """
    get the prediction of the models
    input
        model: model
        test_dataset: test_data
    output
        pred_list: prediction
        target_list: true label
        s_list: sensitive_features
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        x_pred = torch.from_numpy(x).float()
        y_pred = torch.from_numpy(y)
        logits = model(x_pred)
        probability = softmax(logits).numpy()
        loss = criterion(logits, y_pred).numpy()
        acc = np.mean(np.argmax(probability, axis=1) == y)

    return probability, loss, acc


def get_performance(model, x, y, s, attr_idx):
    """
    get the prediction performance of the models: accuracy, eogap, dp gap
    input
        model: model
        test_dataset: test_data
    output
        pred_list: prediction
        target_list: true label
        s_list: sensitive_features
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        x_pred = torch.from_numpy(x).float()
        y_pred = torch.from_numpy(y)
        logits = model(x_pred)
        probability = softmax(logits).numpy()
        loss = criterion(logits, y_pred).numpy()
        pred = np.argmax(probability, axis=1)
        acc = np.mean(pred == y)
        pred_acc = pred == y

        tnr_list = []
        tpr_list = []
        ppr_list = []

        converted_s = s[:, attr_idx]

        # converted_s[converted_s > 1] = 0 # 0 non-white, 1 white

        for s_value in np.unique(converted_s):
            if np.mean(converted_s == s_value) > 0.01:
                indexs0 = np.logical_and(y == 0, converted_s == s_value)
                indexs1 = np.logical_and(y == 1, converted_s == s_value)
                ppr_list.append(np.mean(pred[converted_s == s_value]))
                tnr_list.append(np.mean(pred_acc[indexs0]))
                tpr_list.append(np.mean(pred_acc[indexs1]))

    eo_gap = max(max(tnr_list) - min(tnr_list), max(tpr_list) - min(tpr_list))
    eopp_gap = max(tnr_list) - min(tnr_list)
    dp_gap = max(ppr_list) - min(ppr_list)
    return acc, eo_gap, dp_gap, eopp_gap


def get_dataset(test_dataset):
    """
    get the dataset from the dataloader
    input
        test_dataset: test_data
    output
        x_list: dataset
        target_list: true label
        s_list: sensitive_features
    """

    target_list = []
    s_list = []
    x_list = []

    for x, target, s in test_dataset:
        target_list.extend(target.tolist())
        s_list.extend(s.tolist())
        x_list.extend(x.tolist())
    target_list = np.array(target_list)
    s_list = np.array(s_list)
    x_list = np.array(x_list)
    return x_list, target_list, s_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--task", type=str, help="the ML task", default="income")
    args = parser.parse_args()

    random_seed_list = [
        0,
        1,
        2,
        3,
        4,
    ]  # changing the random seed based on the experiment
    res_dict = {
        "client": [],
        "FL": [],
        "Standalone": [],
        "random_seed": [],
        "Centralized": [],
        "FL_eo": [],
        "Standalone_eo": [],
        "Centralized_eo": [],
        "FL_dp": [],
        "Standalone_dp": [],
        "Centralized_dp": [],
        "FL_eopp": [],
        "Standalone_eopp": [],
        "Centralized_eopp": [],
        "attr": [],
    }

    for random_seed in random_seed_list:
        if os.path.isfile(
            "../results/{}/run_{}/data.pkl".format(args.task, random_seed)
        ):
            # load the dataset used by the model
            folder_path = "../results/{}/run_{}".format(args.task, random_seed)
            data_file = "{}/data.pkl".format(folder_path)
            with open(data_file, "rb") as f:  # Python 3: open(..., 'rb')
                dataset = pickle.load(f)

            (
                num_user,
                users,
                train_data_num,
                test_data_num,
                train_data_global,
                test_data_global,
                val_data_global,
                train_data_local_num_dict,
                test_data_local_num_dict,
                train_data_local_dict,
                test_data_local_dict,
                val_data_local_dict,
                class_num,
                unselected_data_local_dict,
            ) = dataset

            fedavg_model = TwoNN(
                census_input_shape_dict[args.task], hidden_shape_dict[args.task], 2
            )
            fedavg_model.load_state_dict(torch.load(f"{folder_path}/fedavg.pt"))
            fedavg_model.eval()

            centra_model = TwoNN(
                census_input_shape_dict[args.task], hidden_shape_dict[args.task], 2
            )
            centra_model.load_state_dict(torch.load(f"{folder_path}/centralized.pt"))
            centra_model.eval()

            with open(f"{folder_path}/standalone.pt", "rb") as f:
                standalone_models = pickle.load(f)

            for client_idx in range(51):
                s_model = TwoNN(
                    census_input_shape_dict[args.task],
                    hidden_shape_dict[args.task],
                    2,
                )
                s_model.load_state_dict(standalone_models[client_idx])
                s_model.eval()

                x, y, s = get_dataset(test_data_local_dict[client_idx])

                if args.task == "income":
                    sensitive_attr_list = [0, 1]
                else:
                    sensitive_attr_list = range(s.shape[1])
                for attr_idx in sensitive_attr_list:
                    res_dict["random_seed"].append(random_seed)
                    res_dict["client"].append(client_idx)

                    (
                        centra_acc,
                        centra_eo_gap,
                        centra_dp_gap,
                        centra_eopp,
                    ) = get_performance(centra_model, x, y, s, attr_idx)
                    (
                        fedavg_acc,
                        fedavg_eo_gap,
                        fedavg_dp_gap,
                        fedavg_eopp,
                    ) = get_performance(fedavg_model, x, y, s, attr_idx)
                    (
                        standalone_acc,
                        standalone_eo_gap,
                        standalone_dp_gap,
                        standalone_eopp,
                    ) = get_performance(s_model, x, y, s, attr_idx)

                    res_dict["Standalone"].append(standalone_acc)
                    res_dict["FL"].append(fedavg_acc)
                    res_dict["Centralized"].append(centra_acc)

                    res_dict["Standalone_eo"].append(standalone_eo_gap)
                    res_dict["FL_eo"].append(fedavg_eo_gap)
                    res_dict["Centralized_eo"].append(centra_eo_gap)

                    res_dict["Standalone_dp"].append(standalone_dp_gap)
                    res_dict["FL_dp"].append(fedavg_dp_gap)
                    res_dict["Centralized_dp"].append(centra_dp_gap)
                    res_dict["Standalone_eopp"].append(standalone_eopp)
                    res_dict["FL_eopp"].append(fedavg_eopp)
                    res_dict["Centralized_eopp"].append(centra_eopp)
                    res_dict["attr"].append(attr_idx)
    res_data = pd.DataFrame.from_dict(res_dict)
    res_data.to_csv(f"saved_information/{args.task}_all_information.csv")
