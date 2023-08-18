import numpy as np
import pandas as pd
import sys
import os
import scipy
import scipy.sparse
import argparse

import parameters

############################################## crystalNumber ################################################
###   #| 00 01 02 03 04 05 06 | 07 08 09 10 11 12 13 14 |#| 15 16 17 18 19 20 21 | 22 23 24 25 26 27 28 29 |#
#############################################################################################################
### 0 #| 70 69 68 67 66 65 64 | 07 06 05 04 03 02 01 00 |#| <--------------- 304 | 247 <---------------240 |#
#R# 1 #| 77 <------------- 71 | 15 <---------------- 08 |#| <------------------- | <---------------------- |#
#I# 2 #| 84 <------------- 78 | 23 <---------------- 16 |#| <------------------- | <---------------------- |#
#N# 3 #| <---------------- 85 | 31 <---------------- 24 |#| <------------------- | <---------------------- |#
#G# 4 #| <------------------- | <------------------- 32 |#| <------------------- | <---------------------- |#
#N# 5 #| <------------------- | <---------------------- |#| <------------------- | <---------------------- |#
#U# 6 #| <------------------- | <---------------------- |#| <------------------- | <---------------------- |#
#M# 7 #| 119 <----------- 113 | 63 <------------------- |#| 359 <--------------- | 303 <------------------ |#
#B# --#|----------------------|-------------------------|#|----------------------|-------------------------|#
#E###########################################################################################################
#R# 8 #| 190 <----------- 184 | 127 <-------------- 120 |#| <--------------- 424 | 367 <-------------- 360 |#
### 9 #| <--------------- 191 | 135 <-------------- 128 |#| <------------------- | <---------------------- |#
###10 #| <------------------- | <------------------ 136 |#| <------------------- | <---------------------- |#

def position_to_crystalID(ringNumber, crystalNumber):
    ModulePositionX = np.floor(crystalNumber / parameters.n_Crystals_Per_Module_X)
    ModulePositionY = np.floor(ringNumber / parameters.n_Crystals_Per_Module_Y)
    PositionOnModuleX = (parameters.n_Crystals_Per_Module_X - (crystalNumber % parameters.n_Crystals_Per_Module_X)) - 1
    PositionOnModuleY = ringNumber % parameters.n_Crystals_Per_Module_Y
    crystalID = (ModulePositionX * (parameters.n_Crystals_Per_Module_X * parameters.n_Rings)) + (ModulePositionY * (parameters.n_Crystals_Per_Module_X * parameters.n_Crystals_Per_Module_Y))

    if PositionOnModuleX < parameters.Split_In_Module:
        crystalID += (parameters.Split_In_Module * PositionOnModuleY) + ModulePositionX
    else:
        crystalID += (parameters.n_Crystals_Per_Module_Y * parameters.Split_In_Module) + (PositionOnModuleY * (parameters.n_Crystals_Per_Module_X - parameters.Split_In_Module)) + (ModulePositionX - parameters.Split_In_Module)

    return crystalID

def k_furthest_neighbours(ringNumber, crystalNumber):
    Xmin = (parameters.n_Crystals_Per_Ring - crystalNumber - parameters.Spread) % parameters.n_Crystals_Per_Ring
    Xmax = ((parameters.n_Crystals_Per_Ring - crystalNumber + parameters.Spread) % parameters.n_Crystals_Per_Ring)
    Ymin = max(parameters.n_Rings - ringNumber - parameters.Spread, 0)
    Ymax = min(parameters.n_Rings - ringNumber + parameters.Spread, parameters.n_Rings - 1)
    geometry = np.zeros(0, dtype=np.int32)
    for posX in range(Xmin, Xmax + 1):
        for posY in range(Ymin, Ymax + 1):
            geometry = np.append(geometry, position_to_crystalID(posY, posX))

    return geometry

class photon:
    def __init__(self, posX, posY, time):
        self.crystalNumber = np.int64(posX)
        self.ringNumber = np.int64(posY)
        #self.time = np.int64(np.random.normal(time, parameters.CoincidenceTimeResolution))
        self.time = time

class decay:
    def __init__(self, time):
        positionX_1 = np.random.randint(0, parameters.n_Crystals_Per_Ring)
        positionY_1 = np.random.randint(0, parameters.n_Rings)
        self.photon_1 = photon(positionX_1, positionY_1, time)
        positionX_2 = np.random.randint(parameters.n_Crystals_Per_Ring - positionX_1 - parameters.Spread, parameters.n_Crystals_Per_Ring - positionX_1 + parameters.Spread) % parameters.n_Crystals_Per_Ring
        positionY_2 = np.random.randint(max(parameters.n_Rings - positionY_1 - parameters.Spread, 0), min(parameters.n_Rings - positionY_1 + parameters.Spread, parameters.n_Rings - 1))
        self.photon_2 = photon(positionX_2, positionY_2, time)
        self.time = time

    def to_numpy(self):
        output = np.array((position_to_crystalID(self.photon_1.ringNumber, self.photon_1.crystalNumber), position_to_crystalID(self.photon_2.ringNumber, self.photon_2.crystalNumber), self.time), dtype=[('crystal1', np.int32), ('crystal2', np.int32), ('time', np.int32)])
        return output

    def photonList(self):
        return [self.photon_1, self.photon_2]

class event:
    def __init__(self, crystalID):
        self.crystal = np.int32(crystalID)
        self.time = 0

    def to_numpy(self):
        output = np.array((self.crystal, self.time), dtype=[('crystal', np.int32), ('time', np.int32)])
        return output

def generate_delays():
    delays = np.random.binomial(n=parameters.delay_max, p=float(parameters.delay_mean/parameters.delay_max), size=parameters.n_Rings * parameters.n_Crystals_Per_Ring)
    #dict_delays = {'crystalID': np.arange(delays.size), 'delay[ps]': delays}
    #df_delays = pd.DataFrame(dict_delays)
    #df_delays.to_csv('Delays_' + str(delay_generation_seed) + '_' + str(simulation_seed) + '.csv',index=False)
    return delays

def bin_projdata(coincidences):
    projdata = np.zeros((parameters.n_Crystals, parameters.n_Crystals), dtype=np.int32)
    for coinc in coincidences:
        projdata[coinc['crystal1'], coinc['crystal2']] += 1
    return projdata




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir', default='/hkfs/work/workspace_haic/scratch/kj3268-PetNet/data', help='directory to save data', type=str)
    parser.add_argument('--nsamples', default=1024000, help='total number of samples to generate', type=int)
    parser.add_argument('--measurementTime', default=1000000, help='simulation timesteps', type=int)
    arguments = parser.parse_args()

    os.chdir(arguments.savedir)

    delay_generation_seed = 0
    for simulation_seed in range(arguments.nsamples):
        np.random.seed(delay_generation_seed)
        delays = generate_delays()
        np.random.seed(simulation_seed)
        events = np.zeros(0, dtype=[('crystal', np.int32), ('time', np.int32)])
        events_geometry = np.zeros(0, dtype=[('crystal', np.int32), ('time', np.int32)])
        coincidences = np.zeros(0, dtype=[('crystal1', np.int32), ('crystal2', np.int32), ('time', np.int32)])
        num_events = max(1, np.random.poisson(parameters.activity))
        decay_times = np.sort(np.random.randint(np.amax(delays), arguments.measurementTime, size=num_events))

        for current_time in decay_times:
            current_decay = decay(current_time) # This converts decay_time from seconds to Picoseconds
            n_Hits = 0
            for current_photon in current_decay.photonList():
                if np.random.random() < parameters.detection_Efficiency:
                    current_event = event(position_to_crystalID(current_photon.ringNumber, current_photon.crystalNumber))
                    current_event.time = current_photon.time - delays[current_event.crystal]
                    events = np.append(events, current_event.to_numpy())
                    for crystal in k_furthest_neighbours(current_photon.ringNumber, current_photon.crystalNumber):
                        events_geometry = np.append(events_geometry, np.array((crystal, current_event.time), dtype=[('crystal', np.int32), ('time', np.int32)]))
                    n_Hits += 1
            if n_Hits == 2:
                coincidences = np.append(coincidences, current_decay.to_numpy())

        projdata = bin_projdata(coincidences)

        events_sparse = scipy.sparse.coo_array((np.ones(len(events['crystal'])), (events['crystal'], events['time'])), shape=(parameters.n_Crystals, arguments.measurementTime))
        events_geometry_sparse = scipy.sparse.coo_array((np.full(len(events_geometry['crystal']), 0.1), (events_geometry['crystal'], events_geometry['time'])), shape=(parameters.n_Crystals, arguments.measurementTime))
        coincs_sparse = scipy.sparse.coo_array((np.ones(2 * len(coincidences['crystal1'])), (np.concatenate([coincidences['crystal1'], coincidences['crystal2']]), np.concatenate([coincidences['time'],coincidences['time']]))), shape=(parameters.n_Crystals, arguments.measurementTime))
        projdata_sparse = scipy.sparse.coo_array(projdata)

        scipy.sparse.save_npz('petsim/Events_' + str(delay_generation_seed) + '_' + str(simulation_seed) + '.npz', events_sparse)
        scipy.sparse.save_npz('petsim_plusgeometry/Events_' + str(delay_generation_seed) + '_' + str(simulation_seed) + '.npz',scipy.sparse.vstack([events_sparse, events_geometry_sparse]))
        scipy.sparse.save_npz('petsim/Coincidences_' + str(delay_generation_seed) + '_' + str(simulation_seed) + '.npz',coincs_sparse)
        scipy.sparse.save_npz('petsim_plusgeometry/Coincidences_' + str(delay_generation_seed) + '_' + str(simulation_seed) + '.npz',coincs_sparse)
        scipy.sparse.save_npz('petsim/Projdata_' + str(delay_generation_seed) + '_' + str(simulation_seed) + '.npz',projdata_sparse)
        scipy.sparse.save_npz('petsim_plusgeometry/Projdata_' + str(delay_generation_seed) + '_' + str(simulation_seed) + '.npz',projdata_sparse)
