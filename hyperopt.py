#!/usr/bin/env python3

from mpi4py import MPI

from propulate import Islands
from propulate.utils import get_default_propagator
# from propulate.propagators import SelectMin, SelectMax


import snntorch as snn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import matplotlib.pyplot as plt
import sys
import itertools
import os
import argparse
from tqdm import tqdm
import random
import time


import simDataSet
import metrics_helper 
import eval_helper  

GPUS_PER_NODE = 4

def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PETNet(nn.Module):
    def __init__(self, num_layers=1, num_inputs = 28*28, num_hidden = 1000, num_outputs = 10, num_steps = 25, beta = 0.95):
        super().__init__()

        self.best_accuracy = 0.0  # Initialize the model's best validation accuracy.

        self.num_layers = num_layers
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.num_steps = num_steps
        self.beta = beta

        # Initialize layers

        self.fc_hidden1 = nn.Linear(self.num_inputs, self.num_hidden)
        self.lif_hidden1 = snn.Leaky(beta=self.beta, learn_beta=True)
        
        self.fc_hidden2 = nn.Linear(self.num_hidden, self.num_hidden)
        self.lif_hidden2 = snn.Leaky(beta=self.beta, learn_beta=True)
     
        self.fc_out = nn.Linear(self.num_hidden, self.num_outputs)
        self.lif_out = snn.Leaky(beta=self.beta, learn_beta=True)

        # Function to calculate Accuracy
        self.val_acc = 0

    def forward(self, x):

        mem_h1 = self.lif_hidden1.init_leaky()
        mem_h2 = self.lif_hidden2.init_leaky()
        mem_out = self.lif_out.init_leaky()

        # Record the final layer
        spkout_rec = []
        memout_rec = []

        for step in range(self.num_steps):
            cur1 = self.fc_hidden1(x[step])
            spk1, mem_h1 = self.lif_hidden1(cur1, mem_h1)
            if self.num_layers == 2 :
                cur2 = self.fc_hidden2(spk1)
                spk2, mem_h2 = self.lif_hidden2(cur2, mem_h2)
                curout = self.fc_out(spk2)
                spk_out, mem_out = self.lif_out(curout, mem_out)
            else:
                curout = self.fc_out(spk1)
                spk_out, mem_out = self.lif_out(curout, mem_out)
    
            spkout_rec.append(spk_out)
            memout_rec.append(mem_out)

        return torch.stack(spkout_rec, dim=0), torch.stack(memout_rec, dim=0)



def get_data_loaders(batch_size):
#def get_dataloaders_ddp(hyperparameters, num_workers=0, validation_fraction=0.1):

    select_label = 'Coincidences'
    data_path = "data/TimeSteps1000/petsim"
    nsamples = 3000
    dataSet = simDataSet.SimDataSet(data_dir=data_path, label=select_label, transform = None)
    dataSet = torch.utils.data.Subset(dataSet, range(0, nsamples))

    train_dataset, val_dataset = torch.utils.data.random_split(dataSet, [0.8, 0.2])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)


    return train_loader, val_loader


def ind_loss(params):
    #######Extract hyperparams from params
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    best_loss = 100
    F1 = 0.0
 
    num_layers = params["numlayers"]
    num_nodes = params["numnodes"]
    batch_size = 128 # fix batch size
    #loss_func = params["lossfunction"]  
    
    lr = params["lr"]    
    #loss_funcs = {"mse_count_loss": metrics_helper.mse_count_loss(), 
    #              "mse_count_timing_loss": metrics_helper.mse_count_timing_loss(weight=lossweight, device=device), 
    #              "mse_count_chamfer_loss_pretty": metrics_helper.mse_count_chamfer_loss_pretty(weight=lossweight, device=device), 
    loss_fn = metrics_helper.mse_count_timing_loss(weight=0.1, device=device)  # Get activation function.


    ########### Training Setup
    set_all_seeds(42)
    epochs = 35

    ########## Model
    model = PETNet(num_layers= num_layers, num_inputs = 240, num_hidden = num_nodes, num_outputs = 240, num_steps = 1000, beta = 0.95)

    ########## Dataloaders
    train_loader, val_loader = get_data_loaders(batch_size)

    ####### Train Model
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    start = time.perf_counter() # Measure time for run

    for epoch in range(epochs):
        model.train()
        
        # Minibatch training loop
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device).permute(2,0,1)
            targets = targets.to(device).permute(2,0,1)

            optimizer.zero_grad()

            # forward pass
            spk_rec, mem_rec = model(data)

            # calculate Loss
            loss = loss_fn(spk_rec, targets)
            loss.backward()
            optimizer.step()

        
        # Test set
        model.eval()
        metrics = torch.zeros(6, device=device)
        val_loss = 0
        with torch.no_grad():
            for batch_idx, (val_data, val_targets) in enumerate(val_loader):
                val_data = val_data.to(device).permute(2,0,1)
                val_targets = val_targets.to(device).permute(2,0,1)

                # Test set forward pass
                val_spk, val_mem = model(val_data)

                # Test set loss
                loss_val = loss_fn(val_spk, val_targets)

                metrics += eval_helper.evaluate_spikes(val_spk, val_targets, delay_tolerance=40)
                
                # Calculate Statistics
                val_loss += loss_val.item()

            #At the end of epoch    
        val_loss=val_loss/(batch_idx+1)

        true_hits=metrics[0]
        pred_hits=metrics[1]
        true_positives=metrics[2]
        false_positives=metrics[3]
        false_negatives=metrics[4]
        precision, recall, f1_score = 0.0, 0.0, 0.0

        if true_positives != 0.0:
            precision = true_positives/(true_positives + false_positives)
            recall = true_positives/(true_positives + false_negatives)
            f1_score = 2.0 * (precision * recall) / (precision + recall)
            F1 = f1_score

        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)


        if epoch == epochs-1:
            elapsed = (time.perf_counter() - start)/60 # Measure training time per epoch.
            print(f'Final Results: True Coinc = {true_hits} | True Positives = {true_positives} | Precision = {precision:.3f} | F1 = {f1_score:.3f} | Val_Loss = {val_loss:.5f}, Best_Loss = {best_loss:.5f} \n With num_layers={num_layers}, num_nodes={num_nodes}, batch_size={batch_size} and lr={lr:.6f} | Time elapsed: {elapsed:.2f} min')

    ###### Return validation accuracy
    
    return -F1.item()


if __name__ == "__main__":
    num_generations = 100
    pop_size = 2 * MPI.COMM_WORLD.size
    limits = {
        "numlayers": (1, 2),
        "numnodes": (120, 480),
        #"batchsize": (16,128),
        "lr": (0.01, 0.0001),
#        "lossfunction": ("mse_count_loss", "mse_count_timing_loss", "mse_count_chamfer_loss_ugly"),
    }
    rng = random.Random(MPI.COMM_WORLD.rank)  # Set up separate random number generator for evolutionary optimizer.
    propagator = get_default_propagator(  # Get default evolutionary operator.
        pop_size=pop_size,  # Breeding population size
        limits=limits,  # Search space
        mate_prob=0.7,  # Crossover probability
        mut_prob=0.4,  # Mutation probability
        random_prob=0.1,  # Random-initialization probability
        rng=rng  # Random number generator for evolutionary optimizer
    )
    islands = Islands(  # Set up island model.
        loss_fn=ind_loss,  # Loss function to optimize
        propagator=propagator,  # Evolutionary operator
        rng=rng,  # Random number generator
        generations=num_generations,  # Number of generations per worker
        num_isles=2,  # Number of islands
        migration_probability=0.9  # Migration probability
    )
    islands.evolve(  # Run evolutionary optimization.
        top_n=1,  # Print top-n best individuals on each island in summary.
        logging_interval=1  # Logging interval
    )