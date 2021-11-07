from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
import torch
import json
import numpy as np
import os
from datasets import load_metric
from sklearn.model_selection import StratifiedKFold
from textda.data_expansion import data_expansion


os.environ["TOKENIZERS_PARALLELISM"] = 'false'
PRETRAIN = 'hfl/rbtl3'
metric = load_metric("f1")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels, average='macro')


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def get_prediction():
    """
    train a model to get estimation of each data point
    """
    all_text, all_label, all_id = [], [], []
    for line in open('../../datasets/cic/train.json', 'r', encoding='utf-8'):
        all_text.append(json.loads(line)['sentence'])
        all_label.append(int(json.loads(line)['label']))
        all_id.append(int(json.loads(line)['id']))
    for line in open('../../datasets/cic/dev.json', 'r', encoding='utf-8'):
        all_text.append(json.loads(line)['sentence'])
        all_label.append(int(json.loads(line)['label']))
        all_id.append(int(json.loads(line)['id']))
    label_text, label_label = [], []
    for line in open('../../datasets/cic/label_data.json', 'r', encoding='utf-8'):
        label_text.append(json.loads(line)['sentence'])
        label_label.append(int(json.loads(line)['label']))

    dev_out = {}
    dev_index = {}
    kf = StratifiedKFold(n_splits=6)
    kf.get_n_splits(all_text, all_label)
    for kf_id, (train_index, test_index) in enumerate(kf.split(all_text, all_label)):
        train_text = [all_text[i] for i in train_index][:] + label_text
        train_label = [all_label[i] for i in train_index][:] + label_label
        dev_text = [all_text[i] for i in test_index]
        dev_label = [all_label[i] for i in test_index]
        dev_index[kf_id] = test_index

        new_train_text = []
        new_train_label = []
        for idx, tmp_text in  enumerate(train_text):
            sen_list = data_expansion(tmp_text, alpha_ri=0.1, alpha_rs=0, num_aug=5)
            new_train_text.extend(sen_list)
            new_train_label.extend([train_label[idx]] * len(sen_list))
        train_text = new_train_text
        train_label = new_train_label

        tokenizer = AutoTokenizer.from_pretrained(PRETRAIN, do_lower_case=True)
        train_encodings = tokenizer(train_text, truncation=True, padding=True, max_length=32)
        val_encodings = tokenizer(dev_text, truncation=True, padding=True, max_length=32)

        train_dataset = MyDataset(train_encodings, train_label)
        val_dataset = MyDataset(val_encodings, dev_label)

        # predict dev
        training_args = TrainingArguments(
            # output directory
            output_dir='../../tmpresults/tmpresult{}'.format(kf_id),
            num_train_epochs=50,              # total number of training epochs
            per_device_train_batch_size=32,  # batch size per device during training
            per_device_eval_batch_size=32,   # batch size for evaluation
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            learning_rate=3e-4 if 'electra' in PRETRAIN else 2e-5,
            weight_decay=0.01,               # strength of weight decay
            logging_dir='../../tmplogs',            # directory for storing logs
            logging_steps=10,
            evaluation_strategy="epoch",
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            PRETRAIN, num_labels=118)
        trainer = Trainer(
            # the instantiated 🤗 Transformers model to be trained
            model=model,
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=val_dataset,             # evaluation dataset
            compute_metrics=compute_metrics,
        )
        trainer.train()
        dev_outputs = trainer.predict(val_dataset).predictions
        dev_out[kf_id] = dev_outputs
    alls = [0] * len(all_label)
    for kfid in range(6):
        for idx, item in enumerate(dev_index[kfid]):
            alls[item-1] = dev_out[kfid][idx]
    outputs = np.array(alls)
    return outputs


if __name__ == '__main__':
    get_prediction()
