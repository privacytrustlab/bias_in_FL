import numpy as np
import torch


def rgb(x, y, z):
    return (x / 255, y / 255, z / 255)


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


def get_performance_per_group(model, x, y, s, attr_idx, is_global=False):
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
    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        x_pred = torch.from_numpy(x).float()
        logits = model(x_pred)
        probability = softmax(logits).numpy()
        pred = np.argmax(probability, axis=1)
        pred_acc = pred == y

        tnr_list = []
        tpr_list = []
        ppr_list = []
        acc_list = []
        converted_s = s[:, attr_idx]
        all_s = []
        for s_value in np.unique(converted_s):
            if np.mean(converted_s == s_value) > 0.01 and is_global is False:
                acc_list.append(np.mean(pred_acc[converted_s == s_value]))
                indexs0 = np.logical_and(y == 0, converted_s == s_value)
                indexs1 = np.logical_and(y == 1, converted_s == s_value)
                ppr_list.append(np.mean(pred[converted_s == s_value]))
                tnr_list.append(np.mean(pred_acc[indexs0]))
                tpr_list.append(np.mean(pred_acc[indexs1]))
                all_s.append(s_value)
            elif is_global:
                acc_list.append(np.mean(pred_acc[converted_s == s_value]))

                indexs0 = np.logical_and(y == 0, converted_s == s_value)
                indexs1 = np.logical_and(y == 1, converted_s == s_value)
                ppr_list.append(np.mean(pred[converted_s == s_value]))
                tnr_list.append(np.mean(pred_acc[indexs0]))
                tpr_list.append(np.mean(pred_acc[indexs1]))
                all_s.append(s_value)
    return ppr_list, tpr_list, tnr_list, all_s, acc_list
