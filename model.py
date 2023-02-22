
import torch


class TwoNN(torch.nn.Module):
    # linear model
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
    