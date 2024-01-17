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
    def __init__(self, data_dir, label):
        """
        Arguments:
            data_dir (string): Directory with all data files.
        """
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


        return features, labels

