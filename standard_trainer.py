import os
import torch
from torch import nn
from fedml.core.alg_frame.client_trainer import ClientTrainer
import numpy as np
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

"""
Standard  local trainer; the parties minimize the loss
"""


class StandardTrainer(ClientTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        epoch_loss = []
        model = self.model

        model.to(device)
        model.train()
        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )
        epoch_loss = []
        for _ in range(args.epochs):
            batch_loss = []
            for x, labels, s in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

    def test_on_the_server(self):
        return False

    def test(self, test_data, device, args):
        model = self.model
        model.to(device)
        model.eval()

        metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}
        criterion = nn.CrossEntropyLoss().to(device)

        target_list = []
        s_list = []
        x_list = []
        pred_list = []

        with torch.no_grad():
            for x, target, s in test_data:
                target_list.extend(target.tolist())
                s_list.extend(s.tolist())
                x_list.extend(x.tolist())

                x = x.to(device)
                target = target.to(device)
                s = s.to(device)
                logits = model(x)
                loss = criterion(logits, target)

                _, predicted = torch.max(logits, -1)

                correct = predicted.eq(target).sum()
                pred_list.extend(predicted.detach().cpu().tolist())
                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                metrics["test_total"] += target.size(0)

        target_list = np.array(target_list)
        s_list = np.array(s_list)
        x_list = np.array(x_list)
        pred_list = np.array(pred_list)
        pred_acc = pred_list == target_list

        ppr_list = []
        tnr_list = []
        tpr_list = []
        converted_s = s_list[:, 1]  # sex, 1 attribute

        for s_value in np.unique(converted_s):
            if np.mean(converted_s == s_value) > 0.01:
                indexs0 = np.logical_and(target_list == 0, converted_s == s_value)
                indexs1 = np.logical_and(target_list == 1, converted_s == s_value)
                ppr_list.append(np.mean(pred_list[converted_s == s_value]))
                tnr_list.append(np.mean(pred_acc[indexs0]))
                tpr_list.append(np.mean(pred_acc[indexs1]))

        eo_gap = max(max(tnr_list) - min(tnr_list), max(tpr_list) - min(tpr_list))
        dp_gap = max(ppr_list) - min(ppr_list)

        metrics["eo_gap"] = eo_gap
        metrics["dp_gap"] = dp_gap

        return metrics
