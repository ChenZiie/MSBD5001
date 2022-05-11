import warnings

warnings.filterwarnings('ignore')
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
from torch import optim
import torch
import torch.nn as nn
import random
from torch.utils.data import Dataset
from torch.utils import data
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, f1_score, precision_score
from tqdm import tqdm
from configure.config import config
from model import Naive_model, Prompt_model,Naive_bert,Prompt_bert
from data_prepare import data_pre, Naive_encoder, Prompt_encoder
import glob


def training(epoch, model, train_iter, optimizer, device):
    num, loss_ = 0, 0.0
    for i, batch in tqdm(enumerate(train_iter)):
        optimizer.zero_grad()
        num += 1
        q, a, idx = batch

        for key in q.keys():
           q[key] = q[key].to(device)
        a = a.to(device)

        logits = model(q)


        #logits = model.softmax(logits)

        loss = model.loss(logits,a)
        loss_ += loss.item()
        with torch.autograd.set_detect_anomaly(True):
            loss.backward()
            optimizer.step()

    print("epoch: " + str(epoch) + " average loss: " + str(loss_ / num))

def testing(model, test_iter, device):
    model.eval()
    label = []
    pred = []
    for i, batch in tqdm(enumerate(test_iter)):
        q, a, idx= batch
        for key in q.keys():
            q[key] = q[key].to(device)

        logits = model(q)

        if config["regression_model"]:
            pred += np.rint(logits.cpu().detach().numpy()).tolist()
            label += a.numpy().tolist()
        else:
            logits = model.softmax(logits)
            pred += torch.argmax(logits, dim=-1).cpu().numpy().tolist()
            label += torch.argmax(a, dim=-1).numpy().tolist()

    # print(label)
    # print(pred)
    # precision = precision_score(label_,pred_) * 100
    acc = accuracy_score(label, pred) * 100
    f1 = f1_score(label, pred, average='macro') * 100

    print('Accuracy: ', acc, ' , F1 score: ',f1)
    return acc, f1


if __name__ == '__main__':
    seed_val = 1024
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    #
    # print('Model: 1.Naive 2.Prompt')
    # idx_model = int(input())


    dataset = pd.read_csv('data/sampling_20w_useful.csv')
    train, test = data_pre(dataset,config['English_Only'])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(config['pretrain_model'])
    if config['use_prompt']:
        train_data = Prompt_encoder(train, tokenizer)
        test_data = Prompt_encoder(test, tokenizer)
    else:
        train_data = Naive_encoder(train, tokenizer)
        test_data = Naive_encoder(test,tokenizer)

    train_iter = data.DataLoader(
        dataset=train_data,
        batch_size=config['train_batch_size'],
        shuffle=True,
        num_workers=2)
    test_iter = data.DataLoader(
        dataset=test_data,
        batch_size=config['test_batch_size'],
        shuffle=True,
        num_workers=2)

    if config['use_prompt']:
        model = Prompt_model()
    else:
        model = Naive_model()
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config["learn_rate"])
    scheduler = StepLR(optimizer, step_size=config["lr_dc_step"], gamma=config["lr_dc"])

    print("Train start")
    maxacc = 0
    maxf1 = 0

    for filename in glob.glob('*.pth'):
        if maxacc < float(filename.split('_')[3]):
            maxacc = float(filename.split('_')[3])
        if maxf1 < float(filename.split('_')[5].split('.')[0]):
            maxf1 = float(filename.split('_')[5].split('.')[0])

    for epoch in range(1, config['train_epoch'] + 1):
        training(epoch, model, train_iter, optimizer, device)
        scheduler.step(epoch=epoch)
        if epoch % 1 == 0:
            print('Test result in epoch', epoch)
            acc, f1 =testing(model, test_iter, device)
            if acc > maxacc:
                 torch.save(model.state_dict(),'Best_model_acc_'+str(acc)+'_f1_'+str(f1)+'.pth')
                 maxacc = acc

            if f1 > maxf1:
                 torch.save(model.state_dict(),'Best_model_acc_'+str(acc)+'_f1_'+str(f1)+'.pth')
                 maxf1 = f1

