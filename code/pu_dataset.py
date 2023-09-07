import torch
from torch.utils import data
import pandas as pd
import numpy as np

class PUdata():

    def __init__(self,csv_file):
        if isinstance(csv_file, pd.DataFrame):
            self.dataset = csv_file
        else:
            self.dataset = pd.read_csv(csv_file)
        self.original_label = self.dataset.loc[self.dataset.label==0,"original_label"]
        self.dataset2 = self.dataset.drop(columns = ["original_label"])
        x_data = self.dataset2.drop(labels=['label'], axis=1)
        x_data = np.array(x_data).reshape(x_data.shape[0], 1, x_data.shape[1])
        self.x_data = torch.from_numpy(x_data) 
        self.y_data = torch.from_numpy(np.array(self.dataset2["label"])) 


    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.dataset.shape[0]
    def get_prior(self):
        pos_examples = len([i for i in self.original_label if i==1])
        return pos_examples/self.original_label.shape[0]


class PNdata():

    def __init__(self,csv_file):
        if isinstance(csv_file, pd.DataFrame):
            self.dataset = csv_file
        else:
            self.dataset = pd.read_csv(csv_file)
        self.labeling = self.dataset["label"]
        x_data = self.dataset.drop(labels=['label'], axis=1)
        x_data = np.array(x_data).reshape(x_data.shape[0], 1, x_data.shape[1])
        self.x_data = torch.from_numpy(x_data) 
        self.y_data = torch.from_numpy(np.array(self.dataset["label"])) 
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.dataset.shape[0]
    def get_prior(self):
        pos_examples = len([i for i in self.labeling if i==1])
        return pos_examples/len(self.labeling)
