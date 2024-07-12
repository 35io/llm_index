from tqdm import tqdm
import torch

from loader.DefaultModelLoader import DefaultModelLoader
from loader.DefaultDataLoader import DefaultDataLoader
from loguru import logger


class MainTask(object):
    _test_dataloader = None
    _train_dataloader = None

    def __init__(self, train_dataloader=None, test_dataloader=None):
        self._train_dataloader = train_dataloader
        self._test_dataloader = test_dataloader

    def predict(self):
        loader = DefaultModelLoader()
        model, tokenizer = loader.load('D:\\tools\\llm\\models\\bert-base-uncased')
        data_loader = DefaultDataLoader()
        data_loader.load('D:\\tools\\llm\\datasets\\SST-2\\train.tsv',
                         'D:\\tools\\llm\\datasets\\SST-2\\test.tsv',
                         'D:\\tools\\llm\\datasets\\SST-2\\test.tsv')
        with torch.no_grad():
            for index, batch_data in tqdm(enumerate(data_loader.get_test_dataloader(16)), desc="inference process"):
                encoded = tokenizer(batch_data[0].decode('utf-8'), truncation=True, max_length=150,
                                    pad_to_max_length=True)
                logger.info(encoded)
