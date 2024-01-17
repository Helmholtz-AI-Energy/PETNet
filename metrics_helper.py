import snntorch as snn
import snntorch.functional as snnf
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn


from chamferdist import ChamferDistance

import numpy as np
import sys
import itertools
import os
from tqdm import tqdm
import random



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