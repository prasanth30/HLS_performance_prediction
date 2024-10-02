from data_utils import Design

import os
import networkx as nx
import json
import numpy as np
import pandas as pd
import rich
from utils import Data_Generator, pragmas, collate_fn_train, collate_fn_test
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import sys
# sys.path.append('../Data/')
# base_path = '/home/karakanaidu/experiment/chip/Data/'
base_path = '/Data/'
dgn = Data_Generator(base_path)

class fold_dataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return {'X':self.data[idx].design, 'y':self.data[idx].y,'kernel_name':self.data[idx].kernel_name,'version':self.data[idx].version}

class train_dataset_kfold:
    """
    Manages KFold split of the dataset and returns PyTorch datasets for
    each fold, containing train and validation sets.
    """
    
    def __init__(self, versions=['v18', 'v20', 'v21'], n_folds=5, random_state=42, filter='None'):
        self.versions = versions
        self.use_v18 = 'v18' in versions
        self.use_v20 = 'v20' in versions
        self.use_v21 = 'v21' in versions
        self.n_folds = n_folds  # Number of folds for v21
        self.random_state = random_state  # Seed for reproducibility
        self.filter = filter
        self.base_path = base_path
        self.all_data = []  # Combined data from v18, v20
        self.v21_data = []  # Data specifically from v21
        self.kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        # Load and prepare data
        self.generate_data()

    def generate_data(self):
        # Gather data from v18 and v20 (if enabled)
        for version in self.versions:
            design_path = os.path.join(self.base_path, "train_data", "data", "designs", version)
            self.load_data_from_version(design_path, version)

        # Gather data from v21 and prepare it for KFold
        v21_design_path = os.path.join(self.base_path, "train_data", "data", "designs", "v21")
        self.load_data_from_version(v21_design_path, 'v21', v21_specific=True)

    def load_data_from_version(self, design_path, version, v21_specific=False):
        """
        Helper method to load data from a specific version's directory.
        """
        for fname in os.listdir(design_path):
            if 'json' not in fname:
                continue
            with open(os.path.join(design_path, fname), 'r') as f:
                design_points = json.load(f)
            kernel_name = fname.split('.')[0]
            if kernel_name == 'stencil':
                kernel_name = 'stencil_stencil2d'
            
            for key, points in design_points.items():
                data = Design(kernel_name, version, points)
                if not data.valid and self.filter == 'valid':
                    continue
                if v21_specific:
                    self.v21_data.append(data)
                else:
                    self.all_data.append(data)
    
    def get_folds(self):
        """
        Returns a list of tuples with (train_dataset, valid_dataset) for each fold.
        """
        folds = []

        # Split v21 data into n_folds using KFold
        v21_folds = list(self.kf.split(self.v21_data))
        
        for fold_idx, (train_idx, valid_idx) in enumerate(v21_folds):
            # Train/validation split for v21
            v21_train_data = [self.v21_data[i] for i in train_idx]
            v21_valid_data = [self.v21_data[i] for i in valid_idx]
            
            # Combine v18, v20 data with v21 training data
            combined_train_data = self.all_data + v21_train_data
            
            # Create FoldDataset for training and validation data
            train_dataset = fold_dataset(combined_train_data)
            valid_dataset = fold_dataset(v21_valid_data)
            
            # Append the fold as a tuple of (train_dataset, valid_dataset)
            folds.append((train_dataset, valid_dataset))
        
        return folds


class test_dataset(Dataset):
    
    def __init__(self,pth:str='./../../Data/test.csv'):
        self.df = pd.read_csv(pth)
        self.length = len(self.df)
    def __len__(self):
        return self.length
    
    def __getitem__(self,idx):
        raw_design = self.df.loc[idx]['designs']
        raw_design_list = raw_design.split('.')
        kernel_name = raw_design_list[1].split('-',1)[1]
        version = raw_design_list[0].split('-',1)[1]
        design = {'kernel_name':kernel_name,'version':version}
        
        for element in raw_design_list[2:]:
            name,val = element.split('-',1)
            design[name] = val
        
        return {
            'idx':idx,'design_raw':raw_design,'design':design}

if __name__ == "__main__":
    dataset_kfold = train_dataset_kfold(versions=['v18', 'v20', 'v21'], n_folds=5)
    folds = dataset_kfold.get_folds()

    for i, (train_data, valid_data) in enumerate(folds):
        print(f"Fold {i + 1}:")
        print(f"Training data size: {len(train_data)}")
        print(f"Validation data size: {len(valid_data)}")
        print(train_data[0])
        # You can create DataLoader for each fold
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn_train)
        valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False, collate_fn=collate_fn_test)
        
        break