import tqdm
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
        model, tokenizer = loader.load('/Users/shannon/Documents/tools/llm/model/bert-base-uncased-SST-2')
        data_loader = DefaultDataLoader()
        data_loader.load('/Users/shannon/Documents/tools/llm/datasets/SST-2/train.tsv',
                         None,
                         '/Users/shannon/Documents/tools/llm/datasets/SST-2/test.tsv')
        with torch.no_grad():
            for batch_data in tqdm(data_loader.get_test_dataloader(16), desc="inference process"):
                encoded = tokenizer(batch_data.numpy().decode('utf-8'), truncation=True, max_length=150,
                                    pad_to_max_length=True)
                logger.info(encoded)
