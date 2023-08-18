import os
import fnmatch
import pandas as pd
import torch
import numpy as np
import torch.utils.data as TUData
import sys
import glob
import scipy

class SimDataSet():
    """Simulation dataset."""
    def __init__(self, data_dir : str, label : str, transform : list[str] = None):
        """
        Arguments:
            data_dir (string): Directory with all data files.
            transform (callable, optional): Currently unused.
        """
        self.transform = transform
        self.label = label
        self.feature_files = sorted(glob.glob(data_dir + '/' + 'Events*.npz'))
        self.label_files = sorted(glob.glob(data_dir + '/' + self.label + '*.npz'))
        self.datadir = data_dir
        #if len(self.feature_files) != len(self.label_files):
        #    sys.exit("Mismatch in number of availeable Files! Events: " + str(len(self.feature_files)) + ", " + self.label + ": "  + str(len(self.label_files)))

    def __len__(self):
        return len(self.feature_files)

    def __getitem__(self, idx):
        features_load = scipy.sparse.load_npz(self.feature_files[idx])
        features = torch.sparse_coo_tensor(np.stack([features_load.row,features_load.col]), features_load.data, features_load.shape, dtype=torch.float32)

        labels_load = scipy.sparse.load_npz(self.label_files[idx])
        labels = torch.sparse_coo_tensor(np.stack([labels_load.row,labels_load.col]), labels_load.data, labels_load.shape, dtype=torch.float32)

        features = features.to_dense()
        labels = labels.to_dense()

        if self.transform != None:
            for operation in self.transform:
                command = operation.split()
                if command[0] == 'Flip':
                        features = torch.flip(features, [0])
                        labels = torch.flip(labels, [0])
                elif command[0] == 'RandomCrop':
                        bounds_lower = np.random.randint(0,features.shape[1]-int(command[1])-1)
                        features = features[:,bounds_lower:bounds_lower+int(command[1])]
                        labels = features[:,bounds_lower:bounds_lower+int(command[1])]
                        print(bounds_lower)
                        print(features.shape)
                else:
                        print("Unknown Transform! " + operation)

        return features, labels

