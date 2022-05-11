import numpy as np
import pandas as pd
from torch import optim
import torch
import re
import torch.nn as nn
import random
from torch.utils.data import Dataset
from torch.utils import data
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
from tqdm import tqdm
from configure.config import config


def onehot(x):
    onehot_vector = []
    for i in range(5):
        if i == x -1:
            onehot_vector.append(1)
        else:
            onehot_vector.append(0)
    return onehot_vector

def check_en_str(string):
    import re
    pattern = re.compile('^[A-Za-z0-9.,:;!?()_*"\' ]+$')
    if pattern.fullmatch(string):
        return True
    else:
        return False

def find_max(x):
    x = x.split(' ')
    max_ = 0
    for i in x:
        if max_<len(i):
            max_ = len(i)
    return max_


def data_pre(dataset, ENonly):
    dataset = dataset[['stars', 'text']]
    dataset = dataset.dropna(how='any', axis=0)

    if ENonly:
        dataset['EG'] = dataset['text'].apply(lambda x: check_en_str(x))
        dataset = dataset[(dataset['EG'] == True)]

    dataset['cnt'] = dataset['text'].apply(lambda x: find_max(x))
    dataset = dataset[(dataset['cnt'] <= config['max_len'] - 20)]

    test = dataset.sample(n=1000)
    # print(len(dataset))
    # dataset = dataset.append(test).drop_duplicates(keep=False)
    # print(len(dataset))
    grouped = dataset.groupby('stars', group_keys=False)
    train = grouped.apply(lambda x: x.sample(3000))

    train['one_hot_stars'] = train['stars'].apply(lambda x: onehot(x))
    test['one_hot_stars'] = test['stars'].apply(lambda x: onehot(x))

    if config['use_prompt']:
        train['text'] = train['text'].apply(lambda x: config['prompt'] + x)
        test['text'] = test['text'].apply(lambda x: config['prompt'] + x)


    train.index = range(0, len(train))
    test.index = range(0, len(test))
    return train, test



class Naive_encoder(Dataset):
    def __init__(self, dataest, tokenizer):
        self.x = dataest['text']
        self.one_hot_stars = list(dataest['one_hot_stars'])
        self.stars = list(dataest['stars'])
        self.config = config
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        data_x = self.tokenizer.encode_plus(
            self.x[idx],
            add_special_tokens=True,
            max_length=self.config['max_len'],
            pad_to_max_length=True,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        data_x = {
            'input_ids': data_x['input_ids'].flatten(),
            'attention_mask': data_x['attention_mask'].flatten(),
        }

        if config["regression_model"]:
            y = torch.tensor(self.stars[idx], dtype=torch.float)
        else:
            y = torch.tensor(self.one_hot_stars[idx], dtype=torch.float)

        return data_x, y, 1

    def __len__(self):
        return len(self.x)

class Prompt_encoder(Dataset):
    def __init__(self, dataest, tokenizer):
        self.x = dataest['text']
        self.y = list(dataest['one_hot_stars'])
        self.config = config
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        data_x = self.tokenizer.encode_plus(
            self.x[idx],
            add_special_tokens=True,
            max_length=self.config['max_len'],
            pad_to_max_length=True,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        data_x = {
            'input_ids': data_x['input_ids'].flatten(),
            'attention_mask': data_x['attention_mask'].flatten(),
        }


        return data_x, torch.tensor(self.y[idx], dtype=torch.float), 1

    def __len__(self):
        return len(self.x)