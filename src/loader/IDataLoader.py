from abc import ABCMeta, abstractmethod


class IDataLoader(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def load(self, path: str):
        pass

    @abstractmethod
    def get_train_dataloader(self):
        pass

    @abstractmethod
    def get_val_dataloader(self):
        pass

    @abstractmethod
    def get_test_dataloader(self):
        pass