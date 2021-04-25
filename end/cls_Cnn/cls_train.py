#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os

from aleph.data.utils import AlephDataDir
from base import MultiLabelProcessor as Processor
from base import Trainer
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
from aleph.data.textclassification import (SequenceClassificationDataset, SequenceClassificationProcessor)
from torch.utils.data.dataloader import DataLoader, default_collate
from torch.utils.data import TensorDataset
from sklearn.metrics import classification_report
import pandas as pd 
from dataloader import *
from utils import *

from sklearn.model_selection import train_test_split
from model_cnn import *
import pickle

df = pickle.load(open('../../end_train_datas/datas_df_total_20210224_online.pkl', 'rb'))
df_yanshou = pickle.load(open('../../end_train_datas/datas_df_yanshou_20210224_online.pkl', 'rb'))
print(df.columns)

num_classes = len(set(df.label.tolist()))
num_classes = 11

print('num_cls: ', num_classes)
print(df.groupby('label').count())
print(df_yanshou.groupby('label').count())

# df = df.iloc[:1000]
# df_yanshou = df.iloc[:1000]
df_train, df_val = train_test_split(df, test_size=0.1, random_state=2021)
# df_train, df_test = train_test_split(df_train, test_size=0.1, random_state=2021)
df_test = df.iloc[:100]
labels = set(df.label.tolist())
print(labels)
print('train,val,test,yanshou - len: ', len(df_train), len(df_val), len(df_test), len(df_yanshou))

'''
def transfer_datas(df, num_classes):
    x = df.features_padding.tolist()
    x = torch.tensor(x)
    # x = TensorDataset(x)

    tmp_labels = df.label.tolist()
    
    y = []
    for idx, i in enumerate(tmp_labels):
        # print(idx, i)
        tmp = [0 for a in range(num_classes)]
        tmp[i] = 1

        y.append(tmp)
    
    y = torch.tensor(y)
    return SequenceClassificationDataset(x, y)

'''
# for cross_entropy
def transfer_datas(df, num_classes):
    x = df.features_padding.tolist()
    x = torch.tensor(x)
    # x = TensorDataset(x)

    tmp_labels = df.label.tolist()
    
    y = torch.tensor(tmp_labels)
    return SequenceClassificationDataset(x, y)
# '''

# train_dataset = Dataset_end(df_train, num_classes)
# val_dataset = Dataset_end(df_val, num_classes)
# test_dataset = Dataset_end(df_test, num_classes)

train_dataset = transfer_datas(df_train, num_classes)
val_dataset = transfer_datas(df_val, num_classes)
test_dataset = transfer_datas(df_test, num_classes)
yanshou_dataset = transfer_datas(df_yanshou, num_classes)

print('df_test: ', len(df_test))
print('example: ', train_dataset[0])


model = TextCnn(num_classes=num_classes)

# dataset
data_dir = '../../end_train_datas/'
save_data_dir = os.path.join(data_dir, 'processed')

save_dir = '../../end_train_datas/models/cls_Cnn_20210301_online/'
checkpoint_dir = os.path.join(save_dir, 'checkpoints')
model_save_dir = os.path.join(save_dir, 'saved')
tblog_dir = os.path.join(save_dir, 'runs')

os.makedirs(save_data_dir,exist_ok = True)
os.makedirs(save_dir,exist_ok = True)


# criterion,optimizer,lr_scheduler
batch_size = 128


param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

optimizer_grouped_parameters = [
    {
        'params':
        [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)],
        'weight_decay':
        0.01
    },
    {
        'params':
        [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)],
        'weight_decay':
        0.0
    }
]

lr = 2e-5
num_warmup_steps = 200
num_training_steps = 20000
optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps)

runner = Trainer(model=model, trainset=train_dataset,
                         valset=val_dataset, train_batch_size=batch_size,
                         checkpoint_dir=checkpoint_dir, save_dir=model_save_dir,
                         tblog_dir=tblog_dir, tblog_freq=100,
                         optimizer=optimizer, lr_scheduler=lr_scheduler,
                         checkpoint_freq = 500,
                         val_batch_size=batch_size, max_epochs=50, device='2')

# runner.run()

model_dir = '../../end_train_datas/models/cls_Cnn_20210301_online/'

model_path = os.path.join(model_dir, 'model.pt')
best_model_path = os.path.join(model_dir, 'best.pt')

model = torch.load(model_path, map_location='cpu')
model.load_state_dict(torch.load(best_model_path, map_location='cpu'))

device = torch.device('cpu')
# device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)


def evaluate(model, test_dataset, threshold=0.5, only_one=True):
    from torch.utils.data import DataLoader
    from sklearn.metrics import classification_report, confusion_matrix
    data_loader = DataLoader(test_dataset,
                             batch_size=32,
                             shuffle=False)
    print('start evaluate!')
    import time
    tic = time.time()
    true, pred, proba = [], [], []
    for step, (inputs, labels) in enumerate(data_loader):
        inputs = torch.tensor(inputs, dtype=torch.float).to(device)

        labels = labels.type(torch.float)
        labels = labels.to(device)
        
        outputs = model(inputs)
        # _, predicted = torch.max(outputs.data, 1)
        tmp_proba = torch.softmax(outputs.data, dim=1)
        outputs = outputs.sigmoid()
        predicted = (outputs > threshold).type(torch.long)

        proba.extend(tmp_proba.cpu().tolist())
        true.extend(labels.cpu().tolist())
        pred.extend(predicted.cpu().tolist())
        '''
        wrong = []
        for i in range(len(true)):
            if pred[i] != true[i]:
                wrong.append(1)
            else:
                wrong.append(0)
                '''
    toc = time.time()
    time_cost = toc - tic

    if only_one:
        pass
    # print('true: ', true, 'pred: ', pred)

    print('time_cost: ', time_cost * 1000, time_cost / len(true) * 1000)
    # print('clf report: \n', classification_report(true, pred))
    # print('confusion_matrix: \n', confusion_matrix(true, pred))
    
    return pred, proba

import time
tic = time.time()
pred, proba = evaluate(model, yanshou_dataset)
toc = time.time()
print('time cost: ', toc-tic, (toc-tic)/len(yanshou_dataset))


df_yanshou['pred'] = pred
# df_test['wrong'] = wrong
df_yanshou['proba'] = proba
pickle.dump(df_yanshou, open('./df_yanshou.pkl', 'wb'))
# df_test.to_csv('./df_test.csv', index=False)

# df_yanshou = df_yanshou[['_id', 'text', 'label', 'raw_label_online', 'hit_time', 'true', 'proba', 'pred', 'duration']]
# df_yanshou.to_csv('./df_yanshou.csv', index=False)

