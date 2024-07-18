from .IModelLoader import IModelLoader
from transformers import PreTrainedModel, AutoTokenizer, AutoModelForSequenceClassification


class DefaultModelLoader(IModelLoader):
    def load(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        return model, tokenizer