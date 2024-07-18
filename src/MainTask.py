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
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        loader = DefaultModelLoader()
        model, tokenizer = loader.load('/Users/shannon/Documents/tools/llm/models/bert-base-uncased-SST-2')
        model.to(device)
        data_loader = DefaultDataLoader(tokenizer)
        data_loader.load('/Users/shannon/Documents/tools/llm/datasets/SST2/train.tsv',
                         '/Users/shannon/Documents/tools/llm/datasets/SST2/test.tsv',
                         '/Users/shannon/Documents/tools/llm/datasets/SST2/test.tsv')

        with torch.no_grad():
            for index, batch_data in tqdm(enumerate(data_loader.get_test_dataloader(16)), desc="inference process"):
                labels = batch_data[1]
                input_ids = batch_data[0]['input_ids']
                input_ids = input_ids.to(device)
                attention_mask = batch_data[0]['attention_mask']
                attention_mask = attention_mask.to(device)
                token_type_ids = batch_data[0]['token_type_ids']
                token_type_ids = token_type_ids.to(device)
                outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
                predict_label = torch.argmax(outputs.logits, dim=-1)
                logger.info(predict_label)
