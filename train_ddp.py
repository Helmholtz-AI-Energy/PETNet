# imports
import torch
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

from models import PETNet, LSTM
from simDataSet import SimDataSet

import metrics_helper 
from evaluate import evaluate_spikes

def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(state, is_best, filename='.pth.tar'):
    torch.save(state, 'checkpoint_' + filename)
    if is_best:
        torch.save(state, 'model_best_' + filename)

def get_dataloaders_ddp(hyperparameters, num_workers=0, validation_fraction=0.1):
    
    # Define a transform
    select_label = hyperparameters.label
    dataSet = SimDataSet(data_dir=hyperparameters.datapath, label=select_label)
    if hyperparameters.nsamples != -1:
        dataSet = torch.utils.data.Subset(dataSet, range(0, hyperparameters.nsamples))

    total = len(dataSet)
    num = int(validation_fraction * total)
    train_indices = torch.arange(0, total - num)
    valid_indices = torch.arange(total - num, total)

    train_dataset = torch.utils.data.Subset(dataSet, train_indices)
    valid_dataset = torch.utils.data.Subset(dataSet, valid_indices)
        
    train_sampler = torch.utils.data.distributed.DistributedSampler(
                        train_dataset, 
                        num_replicas=torch.distributed.get_world_size(), 
                        rank=torch.distributed.get_rank(), 
                        shuffle=True, 
                        drop_last=True)
    
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
                        valid_dataset, 
                        num_replicas=torch.distributed.get_world_size(), 
                        rank=torch.distributed.get_rank(), 
                        shuffle=True, 
                        drop_last=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=hyperparameters.batch,
                                               drop_last=True,
                                               sampler=train_sampler)
    
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=hyperparameters.batch,
                                               drop_last=True,
                                               sampler=valid_sampler)


    return train_loader, valid_loader

def get_loss_fn(hyperparameters, device):
    if hyperparameters.loss == 'mse_count_loss':
        loss_fn = metrics_helper.mse_count_loss()
        loss_name = "MSE_Count_Loss"
    elif hyperparameters.loss == 'mse_count_timing_loss':
        loss_fn = metrics_helper.mse_count_timing_loss(weight=hyperparameters.lossweight, device=device)
        loss_name = "MSE_Count_Timing_Loss, weight = "+ str(hyperparameters.lossweight)
    elif hyperparameters.loss == 'mse_count_chamfer_loss_pretty':
        loss_fn = metrics_helper.mse_count_chamfer_loss_pretty(weight=hyperparameters.lossweight, device=device)
        loss_name = "MSE_Count_Chamfer_Loss, (pretty), weight = "+ str(hyperparameters.lossweight)
    elif hyperparameters.loss == 'mse_count_chamfer_loss_ugly':
        loss_fn = metrics_helper.mse_count_chamfer_loss_ugly(weight=hyperparameters.lossweight, device=device)
        loss_name = "MSE_Count_Chamfer_Loss, (ugly), weight = "+ str(hyperparameters.lossweight)
    else:
        loss_fn = metrics_helper.mse_count_loss()
        loss_name = "Loss not set; use MSE_Count_Loss (default)"

    return loss_fn,loss_name


def train(hyperparameters: argparse.Namespace):
    # set fixed seeds for reproducible execution
    set_all_seeds(hyperparameters.seed) 
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    ##############################################################################################
    # dataloader arguments

    train_loader, valid_loader = get_dataloaders_ddp(hyperparameters,validation_fraction=0.2)

    ##############################################################################################
    # Network Architecture
    best_F1 = -1
    best_loss=100
    if hyperparameters.earlystopping > 0: 
        early_stopping = hyperparameters.earlystopping
        patience = 0
    else:
        early_stopping = 0
        
    if hyperparameters.modeltype == "PETNet":
        model = PETNet(num_inputs=hyperparameters.innodes,
                  num_hidden=hyperparameters.hidden,
                  num_outputs=hyperparameters.outnodes,
                  num_steps=hyperparameters.timesteps,
                  beta=hyperparameters.constant)
    elif hyperparameters.modeltype == "LSTM":
        model = LSTM(num_inputs=hyperparameters.innodes,
                      num_hidden=hyperparameters.hidden,
                      num_outputs=hyperparameters.outnodes,
                      device=device)
        
    ddp_model = DDP(model.to(device)) # Wrap model with DDP; needs to be on CUDA for nccl backend

    # Define loss function and Optimizer
    loss_fn, loss_name = get_loss_fn(hyperparameters, device)
    if rank == 0: print(f'Using Loss {loss_name}')
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=hyperparameters.lr, betas=(0.9, 0.999))
    ##############################################################################################

    #Initialization
    val_loss = 0
    precision, recall, f1_score = 0.0, 0.0, 0.0
    true_hits, pred_hits, true_positives,false_positives, false_negatives = 0.0, 0.0, 0.0, 0.0, 0.0
    start = time.perf_counter() # Measure time per epoch.


    for epoch in range(hyperparameters.epochs):
        train_loader.sampler.set_epoch(epoch)   
        start_epoch = time.perf_counter() # Measure time per epoch.
    
        train_loss = 0
        val_loss = 0

        ddp_model.train()
        # Minibatch training loop
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device).permute(2,0,1)
            targets = targets.to(device).permute(2,0,1)

            optimizer.zero_grad()
            loss = None
            
            # forward pass
            spk_rec, mem_rec = ddp_model(data)

            # calculate Loss
            loss = loss_fn(spk_rec, targets)
            loss.backward()
            optimizer.step()

            torch.distributed.all_reduce(loss)
            loss /= world_size
    
            #Running Loss of current epoch
            train_loss += loss.item()
            
        train_loss = train_loss/(batch_idx+1)
        
        # Test set
        ddp_model.eval()
        metrics = torch.zeros(5, device=device)

        with torch.no_grad():
            for batch_idx, (val_data, val_targets) in enumerate(valid_loader):
                val_data = val_data.to(device).permute(2,0,1)
                val_targets = val_targets.to(device).permute(2,0,1)

                # Test set forward pass
                val_spk, val_mem = ddp_model(val_data)

                # Test set loss
                loss_val = loss_fn(val_spk, val_targets)
                torch.distributed.all_reduce(loss_val)
                loss_val /= world_size


                metrics += evaluate_spikes(val_spk, val_targets, delay_tolerance=40)
                
                # Calculate Statistics
                val_loss += loss_val.item()

        #At the end of epoch    
        val_loss=val_loss/(batch_idx+1)
        torch.distributed.all_reduce(metrics)

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
            
        if rank == 0:
            print(f'Epoch: {epoch+1}/{hyperparameters.epochs}' 
                f'| Training Loss: {train_loss:.5f}' 
                f'| Validation Loss: {val_loss:.5f}' 
                f'| Best Loss: {best_loss:.5f}'
                f'| Trues: {true_hits} '
                f'| True positives: {true_positives} '
                f'| false positives: {false_positives} '
                f'| false negatives: {false_negatives} '
                f'| Precision: {precision:.3f} '
                f'| F1 Score: {f1_score:.3f} '
                 )
            #print(f"Trues: {true_hits} | Predicted: {pred_hits}")
            #print(f"True positives: {true_positives} | false positives: {false_positives} | false negatives: {false_negatives}")
            #print(f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1 Score: {f1_score:.3f}")
            
            elapsed_epoch = (time.perf_counter() - start_epoch)/60 # Measure training time per epoch.
            print(f'Time elapsed for epoch {epoch}: {elapsed_epoch:.2f} min')

        is_best = val_loss < best_loss 
        best_F1 = max(f1_score, best_F1)
        best_loss=min(val_loss,best_loss)
        if early_stopping and not is_best:
            patience += 1
            if patience > early_stopping:        
                break
        else:
            patience = 0
            if rank == 0:
                save_checkpoint({'epoch': epoch + 1,'state_dict': ddp_model.state_dict(),'best_loss': best_loss,}, is_best, hyperparameters.modelname)
    
    if rank == 0:
        elapsed = (time.perf_counter() - start)/60 # Measure training time per epoch.
        print(f'Stopping in epoch {epoch+1}/{hyperparameters.epochs} \n Final Results with {loss_name}: True Coinc = {true_hits} | True Positives = {true_positives} | False Positives: {false_positives} | False Negatives: {false_negatives}\n Precision = {precision:.3f} | F1 = {f1_score:.3f} | Val_Loss = {val_loss:.5f}, Best_F1 = {best_F1:.5f} \n Time elapsed: {elapsed:.2f} min')
        print('# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # ')


    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', default=64, help='batch size', type=int)
    parser.add_argument('--constant', default=0.95, help='decay rate constant of neurons', type=int)
    parser.add_argument('--datapath', default='./data/petsim', help='data directory', type=str)
    parser.add_argument('--earlystopping', default=-1, help='patience value for early stopping', type=int)
    parser.add_argument('--epochs', default=10, help='number of training epochs', type=int)
    parser.add_argument('--hidden', default=240, help='number of hidden layer nodes', type=int)
    parser.add_argument('--label', default='Coincidences', help='label name', type=str)
    parser.add_argument('--lr', default=5e-4, help='learning rate of the optimizer', type=float)
    parser.add_argument('--loss', default='mse_count_loss', help='loss function to utilize')
    parser.add_argument('--lossweight', default=0.1, help='hyperparameter for lossfunction', type=float)
    parser.add_argument('--modelname', default='', help='savefile name of the model', type=str)
    parser.add_argument('--modeltype', default='PETNet', help='string identifier of the model. Options: LSTM; PETNet', type=str)
    parser.add_argument('--innodes', default=240, help='number of input nodes', type=int)
    parser.add_argument('--nsamples', default=-1, help='number of samples to use for training', type=int)
    parser.add_argument('--outnodes', default=240, help='number of output nodes', type=int)
    parser.add_argument('--seed', default=42, help='constant random seed for reproduction', type=int)
    parser.add_argument('--timesteps', default=1000, help='number of distinct time steps', type=int)

    arguments = parser.parse_args()
    
    world_size = int(os.getenv("SLURM_NTASKS"))  
    rank = int(os.getenv("SLURM_PROCID"))       # Get individual process ID.
    #print(f"Rank {rank} of {world_size}: device count = {torch.cuda.device_count()}")

    if rank == 0:
        if dist.is_available(): print("Distributed package available...[OK]") # Check if distributed package available.
        if dist.is_nccl_available(): print("NCCL backend available...[OK]")   # Check if NCCL backend available.

    address = os.getenv("SLURM_LAUNCH_NODE_IPADDR")
    port = "29500"
    os.environ["MASTER_ADDR"] = address
    os.environ["MASTER_PORT"] = port
    
    # Initialize DDP.
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    if rank == 0:
        if dist.is_initialized(): print("Process group initialized successfully...[OK]") # Check initialization.
        # Check used backend.
        print(dist.get_backend(), "backend used...[OK]")

    train(arguments)
    dist.destroy_process_group()



