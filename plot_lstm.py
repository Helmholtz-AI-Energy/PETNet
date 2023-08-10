# imports
import snntorch as snn
import snntorch.functional as snnf
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
import itertools
import os
import argparse
from tqdm import tqdm
import random

from model_lstm import Net
from simDataSet import SimDataSet


def plot_snn_spikes(spk_in, out_test, out_pred, title, entry, arguments):
  # Generate Plots
  fig, ax = plt.subplots(3, figsize=(100,8), sharex=True,
                        gridspec_kw = {'height_ratios': [1, 1, 1]})

  # Plot input spikes
  splt.raster(spk_in, ax[0], s=40, c="black", marker="1")
  ax[0].set_ylabel("Input Spikes")
  ax[0].set_ylim([0, arguments.nodes])
  ax[0].set_title(title)
  ax[0].set_xlim([0, arguments.timesteps])
  ax[0].grid()

  # Plot hidden layer spikes
  splt.raster(out_test.reshape(arguments.timesteps, -1), ax[1], s = 40, c="green", marker="2")
  ax[1].set_ylabel("Output Target Spikes")
  ax[1].set_ylim([0, arguments.outputs])
  ax[1].set_xlim([0, arguments.timesteps])
  for element in enumerate(out_test.nonzero()):
    ax[1].annotate(str(element[1].cpu().detach().numpy()[1]), element[1].cpu().detach().numpy())
  ax[1].grid()

  # Plot output spikes
  splt.raster(out_pred.reshape(arguments.timesteps, -1), ax[2],s = 40, c="red", marker="2")
  ax[2].set_ylabel("Output Predicted Spikes")
  ax[2].set_xlabel("Timestep [ns]")
  ax[2].set_ylim([0, arguments.outputs])
  ax[2].set_xlim([0, arguments.timesteps])
  #for olomont in enumerate(out_pred.nonzero()):
  #  ax[2].annotate(str(olomont[1].cpu().detach().numpy()[1]), olomont[1].cpu().detach().numpy())
  ax[2].grid()

  try:
    os.mkdir('./' + arguments.savedir)
  except:
    pass

  plt.savefig(arguments.savedir + '/' + arguments.savedir + '_' + str(entry) + '.png')
  plt.close()







if __name__ == "__main__":
    torch.manual_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch', default=1, help='batch size', type=int)
    parser.add_argument('--nsamples', default=20, help='number of samples to use for training', type=int)
    parser.add_argument('-c', '--constant', default=0.95, help='decay rate constant of neurons', type=int)
    parser.add_argument('--timesteps', default=1000, help='number of distinct time steps', type=int)
    parser.add_argument('--savedir', default="plots", help='directory to save the images', type=str)
    parser.add_argument('--nodes', default=240, help='number of input nodes', type=int)
    parser.add_argument('--hiddenlayer', default=241, help='number of hidden layer nodes', type=int)
    parser.add_argument('--outputs', default=241, help='number of output nodes', type=int)
    parser.add_argument('--modelpath', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--datapath', default='./data/petsim', type=str, metavar='PATH', help='path to data directory (default: .data/petsim)')

    arguments = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    net = Net(num_inputs=arguments.nodes,
              num_hidden=arguments.hiddenlayer,
              num_outputs=arguments.outputs,
              device = device)


    if os.path.isfile(arguments.modelpath):
        print("=> loading checkpoint '{}'".format(arguments.modelpath))
        checkpoint = torch.load(arguments.modelpath)
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        net.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(arguments.modelpath, checkpoint['epoch']))


        batch_size = arguments.batch
        data_path = arguments.datapath

        dtype = torch.float

        # Define a transform
        select_label = 'Coincidences'
        transform = None
        test_dataset = SimDataSet(data_dir=data_path, label=select_label, transform = transform)
        if arguments.nsamples != -1:
            test_dataset = torch.utils.data.Subset(test_dataset, range(0, arguments.nsamples))
        n_test_samples = len(test_dataset)

        # Create DataLoaders
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)


        net = net.to(device)

        with torch.no_grad():
            net.eval()
            test_loss = 0
            entry = 0
            for test_data, test_targets in test_loader:
                test_data = test_data.to(device).permute(2,0,1)
                test_targets = test_targets.to(device).permute(2,0,1)
                test_targets_nothit = torch.sum(test_targets, dim = 2)
                test_targets_nothit = torch.where(test_targets_nothit == 0.0, 1.0, 0.0).unsqueeze(2)
                test_targets = torch.cat((test_targets, test_targets_nothit), dim = 2)
                # Test set forward pass
                test_spk = net(test_data)
                print(test_spk[:,0].max())
                print(test_spk[:,0].min())
                test_spk[test_spk<0.5] = 0.0
                test_spk[test_spk>=0.5] = 1.0
                plot_snn_spikes(test_data[:,0], test_targets[:,0], test_spk[:,0], "LSTM", entry, arguments)
                entry += 1

    else:
        print("=> no checkpoint found at '{}'".format(arguments.modelpath))

