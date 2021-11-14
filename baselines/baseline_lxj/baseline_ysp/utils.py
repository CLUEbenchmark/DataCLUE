import json
import os
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.nn import CrossEntropyLoss
from transformers import BertTokenizer, BertModel, BertConfig
from time import time
import numpy as np


def LoadData(filepath, left_idx=None):
    
    idx, label, sentence = [], [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            if left_idx and line['id'] in left_idx:
                continue
            if 'label' in line:
                label.append(int(line['label']))
            else:
                label.append(0)
            idx.append(line['id'])
            sentence.append(line['sentence'])

    return {'idx': idx, 'label': label, 'sentence': sentence}


class MyDataset(Dataset):
    def __init__(self, 
                filepath: str = '', 
                mode: str = 'train',
                tokenizer: str = 'bert-base-chinese',
                left_idx: list = None,
                max_length: int = 128):

        self.filepath = filepath
        data = ['train.json', 'dev.json'] if mode == 'train' else ['test_public.json']
        self.idx = []
        self.label = []
        self.sentence = []

        for file in data:
            tmp = LoadData(os.path.join(filepath, file), left_idx)
            self.idx.extend(tmp['idx'])
            self.label.extend(tmp['label'])
            self.sentence.extend(tmp['sentence'])
        
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)

        t1 = time()
        self.sentence = self.tokenizer(self.sentence)# ,truncation=True, padding=True, max_length=max_length)
        t2 = time()
        print(f'{mode} --- tokenizer cost {t2-t1:2.6f} seconds')

    def __getitem__(self, idx):

        data = {}
        data['input_ids'] = torch.tensor(self.sentence['input_ids'][idx])
        data['attention_mask']  = torch.tensor(self.sentence['attention_mask'][idx])
        data['token_type_ids']  = torch.tensor(self.sentence['token_type_ids'][idx])
        data['label']  = torch.tensor([self.label[idx]])
        data['idx'] = self.idx[idx]

        return data

    def __len__(self):
        return len(self.idx)


def collate_fn(batch):
    data = {}

    for tmp in batch:
        for key in tmp:
            if key not in data:
                data[key] = []
            data[key].append(tmp[key])

    for key in data:
        # NOTE escape the index
        if torch.is_tensor(data[key][0]):
            data[key] = pad_sequence(data[key], batch_first=True, padding_value=0)
    
    return data


class BertForSequenceClassification(nn.Module):
    def __init__(self, pretrain_model, num_labels=2):
        super(BertForSequenceClassification, self).__init__()

        self.num_labels = num_labels
        self.pretrain_model = pretrain_model

        self.config = BertConfig.from_pretrained(pretrain_model)
        self.bert = BertModel.from_pretrained(pretrain_model)

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                label=None,
                **kwargs):

        output = self.bert(
            input_ids=input_ids, 
            token_type_ids=token_type_ids, 
            attention_mask=attention_mask
        )
        pooled_output = self.dropout(output['pooler_output'])
        logits = self.classifier(pooled_output)

        loss, predict = None, {}
        predict['logit'] = logits.detach().cpu().numpy()
        predict['pred'] = torch.argmax(logits, dim=-1).detach().cpu().numpy()

        if label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100, reduce=False)
            loss = loss_fct(logits.view(-1, self.num_labels), label.view(-1))
            predict['acc'] = predict['pred'] == label.view(-1).detach().cpu().numpy()
            predict['loss'] = loss.detach().cpu().numpy()
            loss = loss.mean()
        
        return loss, predict


def Stat(result):
    """ Calculates forgetting statistics per example
    result: dictionary created during training containing 
            loss, accuracy, and missclassification margin 
    Copy From Examples forgetting
    """
    needed_to_learn = {}
    unlearned = {}
    margins = {}
    first_learned = {}

    for idx, tmp in result.items():
        acc = np.array(tmp[1])
        transitions = acc[1:] - acc[:-1]

        # Find all presentations when forgetting occurs
        if len(np.where(transitions == -1)[0]) > 0:
            unlearned[idx] = np.where(transitions == -1)[0] + 2
        else:
            unlearned[idx] = []

        if len(np.where(acc == 0)[0]) > 0:
            needed_to_learn[idx] = np.where(acc == 0)[0][-1] + 1
        else:
            needed_to_learn[idx] = 0

        # Find the misclassication margin for each presentation of the example
        margins[idx] = np.array(tmp[2])

        # Find the presentation at which the example was first learned, 
        # e.g. first presentation when acc is 1
        if len(np.where(acc == 1)[0]) > 0:
            first_learned[idx] = np.where(acc == 1)[0][0]
        else:
            first_learned[idx] = np.nan
    
    # NOTE sort examples by stat

    # Initialize lists
    example_original_order = []
    example_stats = []
    for idx in unlearned.keys():

        # Add current example to lists
        example_original_order.append(idx)
        example_stats.append(0)

        # Get all presentations when current example was forgotten during current training run
        stats = unlearned[idx]

        # If example was never learned during current training run, add max forgetting counts
        if np.isnan(first_learned[idx]):
            example_stats[-1] += len(result[idx][0])
        else:
            example_stats[-1] += len(stats)

    return needed_to_learn, unlearned, margins, first_learned, \
        np.array(example_original_order)[np.argsort(example_stats)], np.sort(example_stats)