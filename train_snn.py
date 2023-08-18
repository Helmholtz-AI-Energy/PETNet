# imports
import snntorch as snn
import snntorch.functional as snnf
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from chamferdist import ChamferDistance

import numpy as np
import sys
import itertools
import os
import argparse
from tqdm import tqdm
import random

from model_snn import Net
from simDataSet import SimDataSet


class mse_count_chamfer_loss_ugly(snnf.LossFunctions):
    def __init__(
        self,
        weight,
        device
    ):
        self.__name__ = "mse_count_chamfer_loss_ugly"
        self.weight = weight
        self.device = device

    def __call__(self, spk_out, targets):
        _, num_steps, num_outputs = self._prediction_check(spk_out)
        count_loss_fn = nn.L1Loss(reduction='sum')
        timing_loss_fn = ChamferDistance()

        spike_count = torch.sum(spk_out, 0)
        spike_count_target = torch.sum(targets, 0)

        #Loss Part 1: MSE Loss of the spike count
        count_loss = count_loss_fn(spike_count, spike_count_target)

        #Loss Part 2: MSE Loss of the time Deviations
        baseline_zero = torch.tensor([[[0.]]], requires_grad=True)
        timing_loss = timing_loss_fn(baseline_zero, baseline_zero)

        for sample in range(spk_out.size(dim=1)):
            for crystal in range(spk_out.size(dim=2)):
                spike_timing = spk_out[:,sample,crystal].nonzero(as_tuple=False).unsqueeze(0).cpu().float()
                spike_timing_target = targets[:,sample,crystal].nonzero(as_tuple=False).unsqueeze(0).cpu().float()
                timing_loss += timing_loss_fn(spike_timing, spike_timing_target)

        return (count_loss + (self.weight * timing_loss)) / num_steps



class mse_count_chamfer_loss_pretty(snnf.LossFunctions):
    def __init__(
        self,
        weight,
        device
    ):
        self.__name__ = "mse_count_chamfer_loss_pretty"
        self.weight = weight
        self.device = device

    def __call__(self, spk_out, targets):
        _, num_steps, num_outputs = self._prediction_check(spk_out)

        #Loss Part 1: MSE Loss of the spike count
        count_loss_fn = nn.L1Loss(reduction='sum')

        spike_count = torch.sum(spk_out, 0)
        spike_count_target = torch.sum(targets, 0)
        count_loss = count_loss_fn(spike_count, spike_count_target)

        #Loss Part 2: chamfer distance
        #Note: The Chamfer Distance expects a 3D Tensor of the form [Batch_size , Samples , sample_dimension] as input.
        timing_loss_fn = ChamferDistance()
        #Loss Part 2.0: spk_out and target are of the form [num_steps, batch_size, num_crystals] and we want [batch_size, num_crystals, num_steps]
        spike_timing = spk_out.permute(1,2,0)
        spike_timing_target = targets.permute(1,2,0)
        spike_timing = spike_timing.cpu()
        spike_timing_target = spike_timing_target.cpu()
        #Loss Part 2.1: As we want to look at the crystals individually, we regard each crystal dimension of each batch as its own batch
        #Note: Now the first dimension should be of size batch_size*crystal_num
        spike_timing = spike_timing.reshape((-1,) + spike_timing.shape[2:])
        spike_timing_target = spike_timing_target.reshape((-1,) + spike_timing_target.shape[2:])
        #Loss Part 2.2: We exchange the 'one' of each spike with its index in the time dimension
        spike_timing = spike_timing * torch.arange(1, spike_timing.shape[1]+1)
        spike_timing_target = spike_timing_target * torch.arange(1, spike_timing_target.shape[1]+1)
        #Loss Part 2.3: For the chamfer-distance to work, the contributions of all zero elements have to be 0. we set them to be  -num_steps, so they are all each others nearest neighbour.
        spike_timing = torch.where(spike_timing != 0, spike_timing, torch.full(spike_timing.size(), -1.0 * num_steps))
        spike_timing_target = torch.where(spike_timing_target != 0, spike_timing_target, torch.full(spike_timing_target.size(), -1.0 * num_steps))
        #Loss Part 2.4: need to add an additional empty dimension, as our 'sample_dimension' size for the chamfer distance is 1.
        spike_timing = spike_timing.unsqueeze(2)
        spike_timing_target = spike_timing_target.unsqueeze(2)
        #Loss Part 2.5: Calculate Chamfer Distance.
        timing_loss = timing_loss_fn(spike_timing, spike_timing_target)

        return (count_loss + (self.weight * timing_loss)) / num_steps

class mse_count_loss(snnf.LossFunctions):
    def __init__(
        self,
    ):
        self.__name__ = "mse_count_loss"

    def __call__(self, spk_out, targets):
        _, num_steps, num_outputs = self._prediction_check(spk_out)
        loss_fn = nn.MSELoss()

        spike_count = torch.sum(spk_out, 0)
        spike_count_target = torch.sum(targets, 0)

        loss = loss_fn(spike_count, spike_count_target)
        return loss / num_steps

class mse_count_timing_loss(snnf.LossFunctions):
    def __init__(
        self,
        weight,
        device
    ):
        self.__name__ = "mse_count_timing_loss"
        self.weight = weight
        self.device = device

    def __call__(self, spk_out, targets):
        _, num_steps, num_outputs = self._prediction_check(spk_out)
        count_loss_fn = nn.L1Loss(reduction='sum')
        timing_loss_fn = nn.MSELoss(reduction='sum')

        spike_count = torch.sum(spk_out, 0)
        spike_count_target = torch.sum(targets, 0)

        #Loss Part 1: MSE Loss of the spike count
        count_loss = count_loss_fn(spike_count, spike_count_target)

        #Loss Part 2: MSE Loss of the time Deviations
        baseline_zero = torch.tensor([0.], requires_grad=True, device=self.device)
        timing_loss = timing_loss_fn(baseline_zero, baseline_zero)

        for sample in range(spk_out.size(dim=1)):
            sample_targets = targets[:,sample,:].nonzero(as_tuple=False)
            for pred_hit in spk_out[:,sample,:].nonzero(as_tuple=False):
                Hits = [target_hit for target_hit in sample_targets if pred_hit[1] == target_hit[1]]
                if len(Hits) != 0:
                    target_hit = Hits[0]
                    if len(Hits) != 1:
                        for hit in Hits:
                            if (hit[0] - pred_hit[0]) < (target_hit[0]- pred_hit[0]):
                                target_hit = hit
                    timing_loss += timing_loss_fn(pred_hit[0].type(torch.float), target_hit[0].type(torch.float)) / (num_steps*num_steps)
                    sample_targets = [target for target in sample_targets if not torch.equal(target,target_hit)]

        return (count_loss + (self.weight * timing_loss)) / num_steps

def save_checkpoint(state, is_best, filename='.pth.tar'):
    torch.save(state, 'checkpoint_' + filename)
    if is_best:
        torch.save(state, 'model_best_' + filename)

def evaluate_spikes(spk_out, targets):
    trues_hit, num_trues, false_hits, hit_accuracy = 0.0, 0.0, 0.0, 0.0

    n_samples = spk_out.size(dim=1)
    for sample in range(spk_out.size(dim=1)):
        trues_hit_ps, false_hits_ps, hit_accuracy_ps = 0.0, 0.0, 0.0 #ps = Persample
        sample_targets = targets[:,sample,:].nonzero(as_tuple=False)
        num_trues += len(sample_targets)
        for pred_hit in spk_out[:,sample,:].nonzero(as_tuple=False):
            Hits = [target_hit for target_hit in sample_targets if pred_hit[1] == target_hit[1]]
            if len(Hits) == 0:
                false_hits_ps += 1.0
            else:
                trues_hit_ps += 1.0
                target_hit = Hits[0]
                if len(Hits) != 1:
                    for hit in Hits:
                        if (hit[0] - pred_hit[0]) < (target_hit[0]- pred_hit[0]):
                            target_hit = hit
                hit_accuracy_ps += np.linalg.norm(pred_hit[0].type(torch.float).cpu()-target_hit[0].type(torch.float).cpu())
                sample_targets = [target for target in sample_targets if not torch.equal(target,target_hit)]

        if (trues_hit_ps%2) == 1:
            trues_hit_ps -= 1

        trues_hit += trues_hit_ps
        hit_accuracy += hit_accuracy_ps
        false_hits += false_hits_ps

    false_hits = false_hits / n_samples
    if num_trues == 0:
        trues_hit = 1
        hit_accuracy = 1
    else:
        trues_hit = trues_hit / num_trues
        hit_accuracy = hit_accuracy / num_trues

    return trues_hit, false_hits, hit_accuracy

def calc_f1(spk_out, targets):
    true_positives, false_positives, false_negatives, total_positives = 0.0, 0.0, 0.0, 0.0

    for sample in range(spk_out.size(dim=1)):
        sample_targets = targets[:,sample,:].nonzero(as_tuple=False)
        prediction_hits = spk_out[:,sample,:].nonzero(as_tuple=False)
        total_positives += len(sample_targets)
        sample_targets_sorted = sample_targets[sample_targets[:,0].argsort(dim=0)]
        coinc_hit = 0
        coinc_missed = False
        for target_hit in sample_targets_sorted:
            Hits = [pred_hit for pred_hit in prediction_hits if target_hit[1] == pred_hit[1]]
            if len(Hits) == 0:
                coinc_missed = True
            else:
                pred_hit = Hits[0]
                if len(Hits) != 1:
                    for hit in Hits:
                        if (hit[0] - target_hit[0]) < (pred_hit[0] - target_hit[0]):
                            pred_hit = hit
                prediction_hits = [pred for pred in prediction_hits if not torch.equal(pred,pred_hit)]

            if coinc_hit == 1:
                if coinc_missed == False:
                    true_positives += 1
                else:
                    false_negatives += 1

                coinc_hit = 0
                coinc_missed = False
            else:
                coinc_hit = 1

        false_positives += len(prediction_hits)

    return true_positives, false_positives, false_negatives, total_positives

def train(hyperparameters: argparse.Namespace):
    # set fixed seeds for reproducible execution
    random.seed(hyperparameters.seed)
    np.random.seed(hyperparameters.seed)
    torch.manual_seed(hyperparameters.seed)

    ##############################################################################################
    # dataloader arguments
    data_path = hyperparameters.datapath

    dtype = torch.float
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    # Define a transform
    select_label = 'Coincidences'
    transform = None
    dataSet = SimDataSet(data_dir=data_path, label=select_label, transform = transform)
    if hyperparameters.nsamples != -1:
        dataSet = torch.utils.data.Subset(dataSet, range(0, hyperparameters.nsamples))

    train_dataset, test_dataset = torch.utils.data.random_split(dataSet, [0.9, 0.1])

    n_train_samples = len(train_dataset)
    n_test_samples = len(test_dataset)
    print(n_train_samples)
    print(n_test_samples)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=hyperparameters.batch, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=hyperparameters.batch, shuffle=True, drop_last=True)
    ##############################################################################################
    # Network Architecture
    best_loss = 100
    start_epoch = 0
    net = Net(num_steps=hyperparameters.timesteps,
              num_inputs=hyperparameters.nodes,
              num_hidden=hyperparameters.hiddenlayer,
              num_outputs=hyperparameters.outputs,
              beta=hyperparameters.constant)


    # optionally resume from a checkpoint
    if hyperparameters.resume:
        if os.path.isfile(hyperparameters.resume):
            print("=> loading checkpoint '{}'".format(hyperparameters.resume))
            checkpoint = torch.load(hyperparameters.resume)
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            net.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(hyperparameters.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(hyperparameters.resume))

    # Load the network onto CUDA if available
    net = net.to(device)

    # Define loss function and Optimizer
    if hyperparameters.loss == 'mse_count_loss':
        loss = mse_count_loss()
        print("Using: MSE_Count_Loss")
    elif hyperparameters.loss == 'mse_count_timing_loss':
        loss = mse_count_timing_loss(weight=hyperparameters.lossweight, device=device)
        print("Using: MSE_Count_Timing_Loss, weight = " + str(hyperparameters.lossweight))
    elif hyperparameters.loss == 'mse_count_chamfer_loss_pretty':
        loss = mse_count_chamfer_loss_pretty(weight=hyperparameters.lossweight, device=device)
        print("Using: MSE_Count_Chamfer_Loss, (pretty implementation), weight = " + str(hyperparameters.lossweight))
    elif hyperparameters.loss == 'mse_count_chamfer_loss_ugly':
        loss = mse_count_chamfer_loss_ugly(weight=hyperparameters.lossweight, device=device)
        print("Using: MSE_Count_Chamfer_Loss (ugly implementation), weight = " + str(hyperparameters.lossweight))

    optimizer = torch.optim.Adam(net.parameters(), lr=hyperparameters.lr, betas=(0.9, 0.999))
    ##############################################################################################
    # Main training loop
    # Outer training loop
    for epoch in range(start_epoch, start_epoch + hyperparameters.epochs):
        print(f"epoch: {epoch}")
        net.train()
        trues_hit, false_hits, hit_accuracy = 0.0, 0.0, 0.0
        true_positives, false_positives, false_negatives = 0.0, 0.0, 0.0
        loss_result = 0

        # Minibatch training loop
        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()

            # forward pass
            data = data.permute(2,0,1)
            targets = targets.permute(2,0,1)
            spk_rec, mem_rec = net(data)

            # calculate Loss
            loss_val = loss(spk_rec, targets)
            calc_f1(spk_rec, targets)
            loss_result += loss_val.item()

            # Gradient calculation + weight update
            loss_val.backward()
            optimizer.step()

        # Test set
        with torch.no_grad():
            net.eval()
            test_loss = 0
            for test_data, test_targets in test_loader:
                test_data = test_data.to(device).permute(2,0,1)
                test_targets = test_targets.to(device).permute(2,0,1)

                # Test set forward pass
                test_spk, test_mem = net(test_data)

                # Test set loss
                test_loss += loss(test_spk, test_targets)

                eval_results = evaluate_spikes(test_spk, test_targets)
                trues_hit += eval_results[0]
                false_hits += eval_results[1]
                hit_accuracy += eval_results[2]

                f1_results = calc_f1(test_spk, test_targets)
                true_positives += f1_results[0]
                false_positives += f1_results[1]
                false_negatives += f1_results[2]

        n_batches = n_test_samples/hyperparameters.batch
        if true_positives == 0:
            print("No True Positives!")
            print(f"False_positives: {false_positives}" )
            print(f"False_negatives: {false_negatives}" )
            precision = 0
            recall = 0
            f1_score = 0
        else:
            precision = true_positives/(true_positives + false_positives)
            recall = true_positives/(true_positives + false_negatives)
            f1_score = 2.0 * (precision * recall) / (precision + recall)

        print(f"\t train loss: {loss_result/n_train_samples}")
        print(f"\t test loss: {test_loss/n_test_samples}")
        print(f"\t test accuracy; Trues: {trues_hit/n_batches} \t False: {false_hits/n_test_samples} \t Acc: {hit_accuracy/n_test_samples}")
        print(f"\t Precision: {precision} \t Recall: {recall} \t F1 Score: {f1_score}")

        is_best = test_loss < best_loss
        best_loss = min(test_loss, best_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'best_loss': best_loss,
            }, is_best, hyperparameters.modelname)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', default=64, help='batch size', type=int)
    parser.add_argument('--nsamples', default=-1, help='number of samples to use for training', type=int)
    parser.add_argument('--constant', default=0.95, help='decay rate constant of neurons', type=int)
    parser.add_argument('--epochs', default=10, help='number of training epochs', type=int)
    parser.add_argument('--lr', default=5e-4, help='learning rate of the optimizer', type=float)
    parser.add_argument('--loss', default='mse_count_loss', help='loss function to utilize, options: mse_count_loss, mse_count_timing_loss')
    parser.add_argument('--lossweight', default=0.1, help='hyperparameter for mse_count_timing_loss lossfunction', type=float)
    parser.add_argument('--seed', default=42, help='constant random seed for reproduction', type=int)
    parser.add_argument('--timesteps', default=1000, help='number of distinct time steps', type=int)
    parser.add_argument('--datapath', default='./data/petsim', help='data directory', type=str)
    parser.add_argument('--modelname', default='', help='savefile name of the model', type=str)
    parser.add_argument('--nodes', default=240, help='number of input nodes', type=int)
    parser.add_argument('--hiddenlayer', default=240, help='number of hidden layer nodes', type=int)
    parser.add_argument('--outputs', default=240, help='number of output nodes', type=int)
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

    arguments = parser.parse_args()
    train(arguments)

