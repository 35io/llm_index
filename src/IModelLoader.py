from abc import ABCMeta, abstractmethod


class IModelLoader(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def load(self, path: str):
        pass
