import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import sys
import itertools
import os
import argparse
from tqdm import tqdm
import random
import time

from models import LSTM, PETNet, SCW
from simDataSet import SimDataSet

import metrics_helper 

def evaluate_spikes(spk_out, targets, delay_tolerance = 0):
    matches = []
    #timing_offset = torch.zeros(delay_tolerance+1, device=spk_out.device)    
    
    # Direct comparison of spk_out and targets (hits without delay)
    matches.append(targets*spk_out)
    #timing_offset[0] += (targets*spk_out).sum()
    # Remove hits that were already accounted for
    targets_reduced = targets - targets*spk_out
    spk_reduced = spk_out - targets*spk_out
    # comparison in interval [-delay_tolerance, 0]
    for i in range(1, delay_tolerance+1):
        hits_plus = targets_reduced[i:, :,:]*spk_reduced[:-i, :, :]
        pad = (0,0,0,0,i,0)
        matches.append(torch.nn.functional.pad(hits_plus, pad, "constant", 0))    
    
        # Remove hits that were already accounted for
        #timing_offset[i] += (targets_reduced[i:, :,:]*spk_reduced[:-i, :, :]).sum()
        targets_reduced[i:, :,:] = targets_reduced[i:, :,:] - hits_plus
        spk_reduced[:-i, :, :] = spk_reduced[:-i, :, :] - hits_plus
    
        hits_minus = targets_reduced[:-i, :,:]*spk_reduced[i:, :, :]
        pad = (0,0,0,0,0,i)
        matches.append(torch.nn.functional.pad(hits_minus, pad, "constant", 0))
    
        # Remove hits that were already accounted for
        #timing_offset[i] += (targets_reduced[:-i, :,:]*spk_reduced[i:, :, :]).sum()
        targets_reduced[:-i, :,:] = targets_reduced[:-i, :,:] - hits_minus
        spk_reduced[i:, :, :] = spk_reduced[i:, :, :] - hits_minus
    
    matching_hits=torch.cat(matches, dim=2)
    
    true_coinc = targets.sum(dim=2)
    true_pos = matching_hits.sum(dim=2)
    false_neg = torch.where((true_coinc - true_pos) > 0, true_coinc - true_pos, 0).sum(dim=0)
    
    true_coinc = true_coinc.sum(dim=0)
    true_pos = true_pos.sum(dim=0)
    
    pred_hits = spk_out.sum(dim=2).sum(dim=0) 
    false_pos = spk_reduced.sum(dim=2).sum(dim=0) 
      
    
    #hit_acc = (timing_offset*torch.arange(delay_tolerance+1, device=timing_offset.device)).sum()/true_pos.sum()
    

    return torch.tensor([true_coinc.sum(), pred_hits.sum(), true_pos.sum(), false_pos.sum(), false_neg.sum()],device=spk_out.device)





def run(hyperparameters: argparse.Namespace):
    # set fixed seeds for reproducible execution
    random.seed(hyperparameters.seed)
    np.random.seed(hyperparameters.seed)
    torch.manual_seed(hyperparameters.seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    ##############################################################################################
    # Crate dataloader
    dataSet = SimDataSet(data_dir=hyperparameters.datapath, label='Coincidences', transform = None)
    dataSet = torch.utils.data.Subset(dataSet, range(hyperparameters.samplestart,hyperparameters.samplestart + hyperparameters.nsamples))

    
    val_loader = DataLoader(dataSet, batch_size=hyperparameters.batch, shuffle=True, drop_last=True)

    # Load model checkpoint
    print("Evaluating model '{}'".format(hyperparameters.model_type))
    if hyperparameters.model_type == "SCW":
        model = SCW(window=8)
    elif hyperparameters.model_type == "PETNet":
        model = PETNet(num_inputs=hyperparameters.innodes,
                  num_hidden=hyperparameters.hidden,
                  num_outputs=hyperparameters.outnodes,
                  num_steps=hyperparameters.timesteps,
                  beta=hyperparameters.constant)

        if device == torch.device("cpu"):
            checkpoint = torch.load(hyperparameters.model_path,map_location=torch.device('cpu'))
        else:    
            checkpoint = torch.load(hyperparameters.model_path)
    
        state_dict = checkpoint['state_dict']
        new_state_dict = OrderedDict()
    
        for key, value in state_dict.items():
            new_key = '.'.join(key.split('.')[1:])
            new_state_dict[new_key] = value

        model.eval()
        model.load_state_dict(new_state_dict)    

        # Load the network onto CUDA if available
        model = model.to(device)
    elif hyperparameters.model_type == "LSTM":
        model = LSTM(num_inputs=hyperparameters.innodes,
              num_hidden=hyperparameters.hidden,
              num_outputs=hyperparameters.outnodes,
              device=device)
        
        if device == torch.device("cpu"):
            checkpoint = torch.load(hyperparameters.model_path,map_location=torch.device('cpu'))
        else:    
            checkpoint = torch.load(hyperparameters.model_path)
    
        state_dict = checkpoint['state_dict']
        new_state_dict = OrderedDict()
    
        for key, value in state_dict.items():
            new_key = '.'.join(key.split('.')[1:])
            new_state_dict[new_key] = value

        model.eval()
        model.load_state_dict(new_state_dict)    

        # Load the network onto CUDA if available
        model = model.to(device)

    else:
        print("Modeltype unkown. Stopping Evaluation")
        return 
    metrics = torch.zeros(5, device=device)
    start = time.perf_counter() # Measure time per epoch.

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(loader):
            data = data.to(device).permute(2,0,1)
            targets = targets.to(device).permute(2,0,1)

            spk, _ = model(data)
            metrics += eval_helper.evaluate_spikes(spk, targets, delay_tolerance=40)
            
    true_hits=metrics[0]
    pred_hits=metrics[1]
    true_positives=metrics[2]
    false_positives=metrics[3]
    false_negatives=metrics[4]


    if true_positives == 0.0:
        precision = 0
        recall = 0
        f1_score = 0
    else:
        precision = true_positives/(true_positives + false_positives)
        recall = true_positives/(true_positives + false_negatives)
        f1_score = 2.0 * (precision * recall) / (precision + recall)
        
    print(f'Trues: {true_hits} '
            f'| True positives: {true_positives} '
            f'| False positives: {false_positives} '
            f'| False negatives: {false_negatives} '
            f'| Precision: {precision:.3f} '
            f'| Recall: {recall:.3f} '
            f'| F1 Score: {f1_score:.3f} '
            )
    elapsed_epoch = (time.perf_counter() - start)/60 # Measure training time per epoch.
    print(f'Time elapsed for {hyperparameters.nsamples} samples with batch size {hyperparameters.batch} : {elapsed_epoch:.2f} min')
    print('# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # ')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeltype', default='PETNet', help='string identifier of the model. Options: LSTM; PETNet, SCW', type=str)
    parser.add_argument('--model_path', default='', type=str)
    parser.add_argument('--batch', default=64, help='batch size', type=int)
    parser.add_argument('--constant', default=0.95, help='decay rate constant of neurons', type=int)
    parser.add_argument('--datapath', default='./data/petsim', help='data directory', type=str)
    parser.add_argument('--hidden', default=240, help='number of hidden layer nodes', type=int)
    parser.add_argument('--innodes', default=240, help='number of input nodes', type=int)
    parser.add_argument('--samplestart', default=0, help='number of samples to use for training', type=int)
    parser.add_argument('--nsamples', default=-1, help='number of samples to use for training', type=int)
    parser.add_argument('--outnodes', default=240, help='number of output nodes', type=int)
    parser.add_argument('--seed', default=42, help='constant random seed for reproduction', type=int)
    parser.add_argument('--timesteps', default=1000, help='number of distinct time steps', type=int)


    arguments = parser.parse_args()
    run(arguments)
