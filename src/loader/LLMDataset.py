from torch.utils.data import Dataset
from typing import List, Tuple, Dict
import torch

class LLMDataset(Dataset):

    def __init__(self, data, data_label):
        self.data = data
        self.label = data_label

    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]

        for key_name in data.keys():
            if not isinstance(data[key_name], dict):
                data[key_name] = torch.tensor(data[key_name]).squeeze()

        # label = torch.tensor(self.labels[item_idx]).squeeze().to(self.args.device)

        return data, torch.tensor(labels)

    def __len__(self):
        return len(self.data)
