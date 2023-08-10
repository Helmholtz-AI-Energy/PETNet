# imports
import snntorch as snn

import torch
import torch.nn as nn


# Define Network
class Net(nn.Module):
    def __init__(self, num_inputs = 28*28, num_hidden = 1000, num_outputs = 10, num_steps = 25, beta = 0.95):
        super().__init__()

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.num_steps = num_steps
        self.beta = beta

        # Initialize layers
        self.fc1 = nn.Linear(self.num_inputs, self.num_hidden)
        self.lif1 = snn.Leaky(beta=self.beta, learn_beta=True)
        self.fc2 = nn.Linear(self.num_hidden, self.num_hidden)
        self.lif2 = snn.Leaky(beta=self.beta, learn_beta=True)
        self.fc3 = nn.Linear(self.num_hidden, self.num_outputs)
        self.lif3 = snn.Leaky(beta=self.beta, learn_beta=True)

    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        # Record the final layer
        spk3_rec = []
        mem3_rec = []

        for step in range(self.num_steps):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3_rec.append(spk2)
            mem3_rec.append(mem2)

        return torch.stack(spk3_rec, dim=0), torch.stack(mem3_rec, dim=0)
