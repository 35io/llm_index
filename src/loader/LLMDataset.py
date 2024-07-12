from torch.utils.data import Dataset


class LLMDataset(Dataset):

    def __init__(self, data, data_label):
        self.data = data
        self.label = data_label

    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    def __len__(self):
        return len(self.data)
