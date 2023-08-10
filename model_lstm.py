# imports
import torch
import torch.nn as nn
from torch.autograd import Variable


# Define Network
class Net(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, device):
        super().__init__()
        self.num_layers = 1 #number of layers
        self.device = device
        self.hidden_size = num_hidden #hidden state
        self.lstm = nn.LSTM(input_size=num_inputs, hidden_size=num_hidden,
                          num_layers=self.num_layers, batch_first=True) #lstm
        self.sigm1 = nn.Sigmoid()

    def forward(self,x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) #hidden state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        out = self.sigm1(output)
        return out

