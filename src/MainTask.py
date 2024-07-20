from tqdm import tqdm
import torch

from loader.DefaultModelLoader import DefaultModelLoader
from loader.DefaultDataLoader import DefaultDataLoader
from loguru import logger
from torch.nn.functional import cross_entropy
import random
import numpy as np
from transformers import get_linear_schedule_with_warmup


class MainTask(object):
    _test_dataloader = None
    _train_dataloader = None

    def set_seed(self, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    def __init__(self, train_dataloader=None, test_dataloader=None):
        self._train_dataloader = train_dataloader
        self._test_dataloader = test_dataloader

    def train(self):
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

        optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

        train_dataloader = data_loader.get_train_dataloader(32)
        epochs = 10
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        for epoch in range(epochs):
            logger.info(f'Epoch {epoch + 1}/{epochs}')
            model.train()
            for index, batch_data in tqdm(enumerate(train_dataloader), desc="train process"):
                labels = batch_data[1]
                labels = labels.to(device)
                input_ids = batch_data[0]['input_ids']
                input_ids = input_ids.to(device)
                attention_mask = batch_data[0]['attention_mask']
                attention_mask = attention_mask.to(device)
                token_type_ids = batch_data[0]['token_type_ids']
                token_type_ids = token_type_ids.to(device)
                outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
                predict_label = torch.argmax(outputs.logits, dim=-1)
                loss = cross_entropy(outputs.logits, labels)
                logger.info(loss)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

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

        correct = 0
        total = 0
        with torch.no_grad():
            for index, batch_data in tqdm(enumerate(data_loader.get_test_dataloader(16)), desc="inference process"):
                labels = batch_data[1]
                labels = labels.to(device)
                input_ids = batch_data[0]['input_ids']
                input_ids = input_ids.to(device)
                attention_mask = batch_data[0]['attention_mask']
                attention_mask = attention_mask.to(device)
                token_type_ids = batch_data[0]['token_type_ids']
                token_type_ids = token_type_ids.to(device)
                outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
                predict_label = torch.argmax(outputs.logits, dim=-1)
                logger.info(predict_label)
                correct += (predict_label == labels).sum().item()
                total += labels.size(0)
        logger.info(f'Accuracy is: {100 * correct / total}')
