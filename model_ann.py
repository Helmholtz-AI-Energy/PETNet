# imports
import torch
import torch.nn as nn


# Define Network
class Net(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):

        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x
