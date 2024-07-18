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
        if torch.cuda.is_available():
            device = torch.device("cuda")

        loader = DefaultModelLoader()
        model, tokenizer = loader.load('D:\\tools\\llm\\models\\bert-base-uncased')
        model.to(device)
        data_loader = DefaultDataLoader(tokenizer)
        data_loader.load('D:\\tools\\llm\\datasets\\SST-2\\train.tsv',
                         'D:\\tools\\llm\\datasets\\SST-2\\test.tsv',
                         'D:\\tools\\llm\\datasets\\SST-2\\test.tsv')

        with torch.no_grad():
            for index, batch_data in tqdm(enumerate(data_loader.get_test_dataloader(16)), desc="inference process"):
                labels = batch_data[1]
                input_ids = batch_data[0]['input_ids']
                input_ids = input_ids.to(device)
                attention_mask = batch_data[0]['attention_mask']
                attention_mask = attention_mask.to(device)
                outputs = model(input_ids, token_type_ids=None, attention_mask=attention_mask)
                logger.info(outputs)
