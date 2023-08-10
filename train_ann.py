# imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import itertools
import os
import argparse
from tqdm import tqdm
import random

from model_ann import Net
from simDataSet import SimDataSet

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

        trues_hit += trues_hit_ps
        hit_accuracy += hit_accuracy_ps
        false_hits += false_hits_ps

    trues_hit = trues_hit / num_trues
    false_hits = false_hits / n_samples
    hit_accuracy = hit_accuracy / num_trues

    return trues_hit, false_hits, hit_accuracy


def train(hyperparameters: argparse.Namespace):
    # set fixed seeds for reproducible execution
    random.seed(hyperparameters.seed)
    np.random.seed(hyperparameters.seed)
    torch.manual_seed(hyperparameters.seed)

    ##############################################################################################
    dtype = torch.float
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    # Define a transform
    select_label = 'Coincidences'
    transform = None
    dataSet = SimDataSet(data_dir=hyperparameters.datapath, label=select_label, transform = transform)
    if hyperparameters.nsamples != -1:
        dataSet = torch.utils.data.Subset(dataSet, range(0, hyperparameters.nsamples))

    train_dataset, test_dataset = torch.utils.data.random_split(dataSet, [0.9, 0.1])
    #train_dataset = SimDataSet(data_dir=data_path, train=True, label=select_label, transform = transform)
    #test_dataset = SimDataSet(data_dir=data_path, train=False, label=select_label, transform = transform)

    n_train_samples = len(train_dataset)
    n_test_samples = len(test_dataset)
    print(n_train_samples)
    print(n_test_samples)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=hyperparameters.batch, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=hyperparameters.batch, shuffle=True, drop_last=True)
    print("Data loaded!")
    ##############################################################################################
    # Network Architecture
    best_loss = 100
    start_epoch = 0
    net = Net(num_inputs=hyperparameters.nodes,
              num_hidden=hyperparameters.hiddenlayer,
              num_outputs=hyperparameters.outputs)
    print("Net Created!")


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

    print("Net sent to device!")
    # Define loss function and Optimizer
    loss = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=hyperparameters.lr, betas=(0.9, 0.999))
    ##############################################################################################
    # Main training loop
    # Outer training loop
    for epoch in range(start_epoch, start_epoch + hyperparameters.epochs):
        print(f"epoch: {epoch}")
        net.train()
        trues_hit, false_hits, hit_accuracy = 0.0, 0.0, 0.0
        loss_result = 0

        # Minibatch training loop
        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()

            # forward pass
            predict = net(data)

            # initialize the loss & sum over time
            loss_val = loss(predict, targets)
            loss_result += loss_val.item()
            print(loss_val.item())

            # Gradient calculation + weight update
            loss_val.backward()
            optimizer.step()

        # Test set
        with torch.no_grad():
            net.eval()
            test_loss = 0
            for test_data, test_targets in test_loader:
                test_data = test_data.to(device)
                test_targets = test_targets.to(device)

                # Test set forward pass
                test_predict = net(test_data)

                # Test set loss
                test_loss += loss(test_predict, test_targets).item()

                eval_results = evaluate_spikes(test_spk, test_targets)
                trues_hit += eval_results[0]
                false_hits += eval_results[1]
                hit_accuracy += eval_results[2]


        print(f"\t train loss: {loss_result/n_train_samples}")
        print(f"\t test loss: {test_loss/n_test_samples}")
        print(f"\t test accuracy; Trues: {1.0 - trues_hit/n_test_samples} \t False: {false_hits/n_test_samples} \t Acc: {hit_accuracy/n_test_samples}")

        is_best = test_loss < best_loss
        best_loss = min(test_loss, best_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'best_loss': best_loss,
            }, is_best, hyperparameters.modelname)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', default=2, help='batch size', type=int)
    parser.add_argument('--epochs', default=10, help='number of training epochs', type=int)
    parser.add_argument('--lr', default=5e-4, help='learning rate of the optimizer', type=float)
    parser.add_argument('--seed', default=42, help='constant random seed for reproduction', type=int)
    parser.add_argument('--datapath', default='./data/petsim', help='data directory', type=str)
    parser.add_argument('--nsamples', default=-1, help='number of samples to use for training', type=int)
    parser.add_argument('--modelname', default='', help='savefile name of the model', type=str)
    parser.add_argument('--nodes', default=240*1000, help='number of input nodes', type=int)
    parser.add_argument('--hiddenlayer', default=240*1000, help='number of hidden layer nodes', type=int)
    parser.add_argument('--outputs', default=240*1000, help='number of output nodes', type=int)
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

    arguments = parser.parse_args()
    train(arguments)

