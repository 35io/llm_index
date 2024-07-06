from .IModelLoader import IModelLoader
from transformers import PreTrainedModel, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

class DefaultModelLoader(IModelLoader):
    _dataloaders = {}

    def load(self, train_path: str, test_path: str = None, val_path: str = None):
        df = pd.read_csv(train_path, delimiter='\t', names=['label', 'sentence'])  # names=[  'sentence','label']
        sentences = df.sentence.values[1:]
        labels = df.label.values[1:]

        # if args.model_architect == 'CLM':
        #     instructions = task_prompt[args.dataset]
        #     for i in range(len(sentences)):
        #         sentences[i] = instructions.format(sentences[i])

        X_train = np.array(sentences)
        y_train = np.array([int(_label) for _label in labels])

        df = pd.read_csv(test_set_file, delimiter='\t', names=['label', 'sentence'])  # names=[  'sentence','label']
        sentences = df.sentence.values[1:]
        labels = df.label.values[1:]

        # if args.model_architect == 'CLM':
        #     instructions = task_prompt[args.dataset]
        #     for i in range(len(sentences)):
        #         sentences[i] = instructions.format(sentences[i])

        X_test = np.array(sentences)
        y_test = np.array([int(_label) for _label in labels])

        train_dst = (X_train, y_train)
        test_dst = (X_test, y_test)

    def get_train_dataloader(self):
        pass

    def get_val_dataloader(self):
        pass

    def get_test_dataloader(self):
        pass