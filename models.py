import snntorch as snn
import torch
import torch.nn as nn

class SCW(nn.Module):
    def __init__(self, window = 8):
        super().__init__()
        self.window = window
        
    def forward(self, x):
        output = torch.zeros(data.size())
        x = x.permute(1,2,0)
        for sample in range(x.size(dim=0)):
            data_numpy = x[sample,].nonzero(as_tuple=False).cpu().numpy()
            data_sorted = data_numpy[data_numpy[:,1].argsort()]
            final_coinc_list = []
            current_coinc_list = []
            for i in range(len(data_sorted)):
                if current_coinc_list:
                    if data_sorted[i][1] - current_coinc_list[-1][1] > self.window:
                        if len(current_coinc_list) == 2:
                            final_coinc_list.append(current_coinc_list[0])
                            final_coinc_list.append(current_coinc_list[1])
    
                        current_coinc_list = []
    
                current_coinc_list.append(data_sorted[i])
    
            if len(current_coinc_list) == 2:
                final_coinc_list.append(current_coinc_list[0])
                final_coinc_list.append(current_coinc_list[1])
    
            for entry in final_coinc_list:
                output[sample,entry[0],entry[1]] = 1
    
        return output,0


class PETNet(nn.Module):
    def __init__(self, num_inputs = 28*28, num_hidden = 1000, num_outputs = 10, num_steps = 25, beta = 0.95):
        super().__init__()

        self.best_accuracy = 0.0  # Initialize the model's best validation accuracy.

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.num_steps = num_steps
        self.beta = beta

        self.fc_hidden1 = nn.Linear(self.num_inputs, self.num_hidden)
        self.lif_hidden1 = snn.Leaky(beta=self.beta, learn_beta=True)
    
        self.fc_out = nn.Linear(self.num_hidden, self.num_outputs)
        self.lif_out = snn.Leaky(beta=self.beta, learn_beta=True)



    def forward(self, x):

        mem_h1 = self.lif_hidden1.init_leaky()
        mem_out = self.lif_out.init_leaky()

        spkout_rec = []
        memout_rec = []

        for step in range(self.num_steps):
            cur1 = self.fc_hidden1(x[step])
            spk1, mem_h1 = self.lif_hidden1(cur1, mem_h1)
            curout = self.fc_out(spk1)
            spk_out, mem_out = self.lif_out(curout, mem_out)
    
            spkout_rec.append(spk_out)
            memout_rec.append(mem_out)

        return torch.stack(spkout_rec, dim=0), torch.stack(memout_rec, dim=0)


class LSTM(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, device):
        super().__init__()
        self.num_layers = 1 
        self.device = device
        self.hidden_size = num_hidden 
        self.lstm = nn.LSTM(input_size=num_inputs, hidden_size=num_hidden,
                          num_layers=self.num_layers, batch_first=True) #lstm
        self.sigm1 = nn.Sigmoid()

    def forward(self,x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) #hidden state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) #internal state

        output, (hn, cn) = self.lstm(x, (h_0, c_0)) 
        out = self.sigm1(output)
        return out,0