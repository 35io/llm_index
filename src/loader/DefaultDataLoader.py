import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from .IDataLoader import IDataLoader
from .LLMDataset import LLMDataset


class DefaultDataLoader(IDataLoader):
    _dataloaders = {}
    _train_dst = None
    _test_dst = None

    def load(self, train_path: str, test_path: str = None, val_path: str = None):
        df = pd.read_csv(train_path, delimiter='\t', names=['label', 'sentence'])
        sentences = df.sentence.values[1:]
        labels = df.label.values[1:]

        X_train = np.array(sentences)
        y_train = np.array([int(_label) for _label in labels])

        df = pd.read_csv(test_path, delimiter='\t', names=['label', 'sentence'])
        sentences = df.sentence.values[1:]
        labels = df.label.values[1:]

        X_test = np.array(sentences)
        y_test = np.array([int(_label) for _label in labels])

        self._train_dst = (X_train, y_train)
        self._test_dst = (X_test, y_test)

    def get_train_dataloader(self, batch_size):
        dataset = LLMDataset(self._train_dst[0], self._train_dst[1])
        return DataLoader(dataset=dataset, batch_size=batch_size)

    def get_val_dataloader(self, batch_size):
        pass

    def get_test_dataloader(self, batch_size):
        dataset = LLMDataset(self._test_dst[0], self._test_dst[1])
        return DataLoader(dataset=dataset, batch_size=batch_size)
