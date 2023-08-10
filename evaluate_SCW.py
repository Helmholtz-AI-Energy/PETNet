# imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import itertools
import os
import argparse
import random

from simDataSet import SimDataSet

def evaluate_spikes(spk_out, targets, test_data):
    trues_hit, num_trues, false_hits, hit_accuracy = 0.0, 0.0, 0.0, 0.0

    n_samples = spk_out.size(dim=1)
    for sample in range(spk_out.size(dim=1)):
        trues_hit_ps, false_hits_ps, hit_accuracy_ps = 0.0, 0.0, 0.0 #ps = Persample
        sample_targets = targets[:,sample,:].nonzero(as_tuple=False)
        #if sample_targets.size() != spk_out[:,sample,:].nonzero(as_tuple=False).size():
            #print("NEXT!")
            #print(test_data[:,sample,:].nonzero(as_tuple=False))
            #print(sample_targets)
            #print(spk_out[:,sample,:].nonzero(as_tuple=False))
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

        trues_hit += trues_hit_ps
        hit_accuracy += hit_accuracy_ps
        false_hits += false_hits_ps

    trues_hit = trues_hit / num_trues
    false_hits = false_hits / n_samples
    hit_accuracy = hit_accuracy / num_trues

    return trues_hit, false_hits, hit_accuracy

def do_scw_sorting(data):
    window = 8
    output = torch.zeros(data.size())
    for sample in range(data.size(dim=0)):
        data_numpy = data[sample,].nonzero(as_tuple=False).cpu().numpy()
        data_sorted = data_numpy[data_numpy[:,1].argsort()]
        final_coinc_list = []
        current_coinc_list = []
        for i in range(len(data_sorted)):
            if current_coinc_list:
                if data_sorted[i][1] - current_coinc_list[-1][1] > window:
                    if len(current_coinc_list) == 2:
                        final_coinc_list.append(current_coinc_list[0])
                        final_coinc_list.append(current_coinc_list[1])

                    current_coinc_list = []

            current_coinc_list.append(data_sorted[i])

        if len(current_coinc_list) == 2:
            final_coinc_list.append(current_coinc_list[0])
            final_coinc_list.append(current_coinc_list[1])

        #print("NEXT!")
        #print(data_sorted)
        #print(final_coinc_list)

        for entry in final_coinc_list:
            output[sample,entry[0],entry[1]] = 1

    return output


def evaluate(hyperparameters: argparse.Namespace):
    random.seed(hyperparameters.seed)
    np.random.seed(hyperparameters.seed)

    dtype = torch.float
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    # Define a transform
    select_label = 'Coincidences'
    transform = None
    dataSet = SimDataSet(data_dir=hyperparameters.datapath, label=select_label, transform = transform)
    train_dataset, test_dataset = torch.utils.data.random_split(dataSet, [0.9, 0.1])

    n_train_samples = len(train_dataset)
    n_test_samples = len(test_dataset)
    print(n_train_samples)
    print(n_test_samples)

    train_loader = DataLoader(train_dataset, batch_size=hyperparameters.batch, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=hyperparameters.batch, shuffle=True, drop_last=True)

    trues_hit, false_hits, hit_accuracy = 0.0, 0.0, 0.0
    loss_result = 0
    test_loss = 0

    for test_data, test_targets in test_loader:
        test_data = test_data.to(device)
        test_targets = test_targets.to(device)

        # Test set forward pass
        test_spk = do_scw_sorting(test_data)
        test_spk = test_spk.permute(2,0,1)
        test_targets = test_targets.permute(2,0,1)
        # Test set loss
        eval_results = evaluate_spikes(test_spk, test_targets, test_data.permute(2,0,1))
        trues_hit += eval_results[0]
        false_hits += eval_results[1]
        hit_accuracy += eval_results[2]


    n_batches = n_test_samples/hyperparameters.batch
    print(f"\t test accuracy; Trues: {trues_hit/n_batches} \t False: {false_hits/n_test_samples} \t Acc: {hit_accuracy/n_test_samples}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', default=64, help='batch size', type=int)
    parser.add_argument('--datapath', default='/hkfs/work/workspace_haic/scratch/kj3268-PetNet/data/petsim', help='data directory', type=str)
    parser.add_argument('--seed', default=42, help='constant random seed for reproduction', type=int)
    parser.add_argument('--outname', default='', help='savefile name', type=str)

    arguments = parser.parse_args()
    evaluate(arguments)

