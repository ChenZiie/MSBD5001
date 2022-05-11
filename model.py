import torch
import torch.nn as nn
from torch.nn import Flatten
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import random
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
from configure.config import config


class Naive_model(nn.Module):
    def __init__(self):
        super(Naive_model, self).__init__()
        self.bert = Naive_bert()
        # self.dropout = nn.Dropout(config['dropout'])
        self.linear = nn.Linear(config["seq_feature_dim"], 5)
        self.regression = nn.Linear(config["seq_feature_dim"], 1)
        # self.relu = nn.ReLU()
        self.batch_size = config['train_batch_size']

        self.softmax = nn.Softmax(dim=1)
        self.criterion = nn.CrossEntropyLoss()
        self.criterion1 = nn.MSELoss()

    def forward(self, X_feat):
        logits = self.bert(X_feat)
        # logits = self.dropout(logits)
        if config['regression_model']:
            logits = self.regression(logits)
            logits = logits*torch.full_like(logits,1/250)+torch.ones_like(logits)
        else:
            logits = self.linear(logits)
        # logits = self.relu(logits)
        return logits

    def loss(self, logits, lables):
        if config['regression_model']:
            l = self.criterion1(logits,lables)
        else:
            l = self.criterion(logits,lables)
        return l


class Naive_bert(nn.Module):
    def __init__(self):
        super(Naive_bert, self).__init__()
        self.bert = AutoModel.from_pretrained(config["pretrain_model"])

    def mean_pooling(self, output, attention_mask):
        token_embeddings = output  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, x):
        output = self.bert(**x)

        sentence_embeddings = self.mean_pooling(output[0], x['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings



class Prompt_model(nn.Module):
    def __init__(self):
        super(Prompt_model, self).__init__()
        self.bert = Prompt_bert()
        self.batch_size = config['train_batch_size']

        self.mlp = nn.Sequential(
            nn.Linear(config["seq_feature_dim"], 5))
        self.softmax = nn.Softmax(dim=1)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, X_feat):
        logits = self.bert(X_feat)
        logits = self.mlp(logits)
        return logits

    def loss(self, logits, lables):
        return self.criterion(logits,lables)


class Prompt_bert(nn.Module):
    def __init__(self):
        super(Prompt_bert, self).__init__()
        self.bert = AutoModel.from_pretrained(config["pretrain_model"])


    def mean_pooling(self, output, attention_mask):
        token_embeddings = output  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, x):
        output = self.bert(**x)[0]

        return output[:,config['mask_location']]