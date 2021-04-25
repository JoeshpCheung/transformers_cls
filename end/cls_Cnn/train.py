#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os

from aleph.data.utils import AlephDataDir
from base import MultiLabelProcessor as Processor
from base import Trainer
from aleph.model.textclassification.bert import BertSequenceClassification# , BertForSequenceClassification_AH2
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
from transformers import BertTokenizer
from aleph.data.textclassification import (SequenceClassificationDataset, SequenceClassificationProcessor)
from torch.utils.data.dataloader import DataLoader, default_collate
from torch.utils.data import TensorDataset
import utils
import antiHarassment_api
from sklearn.metrics import classification_report
import pandas as pd 

max_length = 256

# chinese_roberta_wwm_ext_pytorch, wwn-bert-tiny_Fdomai

tokenizer_dir='/data/aleph_data/pretrained/wwn-bert-tiny_Fdomain'
tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)
bert_model_dir='/data/aleph_data/pretrained/wwn-bert-tiny_Fdomain'

def transfer_datas(datas, tokenizer=tokenizer, max_length=max_length):
    input_ids = []
    attention_mask = []
    token_type_ids = []
    y = []
    for data in datas:
        tmp_speaker = data.get('speaker')
        # if tmp_speaker == 'right' or tmp_speaker == 'robot':
        #     continue
        processed = tokenizer(data.get('text_a'), 
                              data.get('text_b'), 
                              max_length=max_length, 
                              padding='max_length', 
                              truncation=True,
                              )
        input_ids += [processed['input_ids']]
        attention_mask += [processed['attention_mask']]
        token_type_ids += [processed['token_type_ids']]
        y.append(data.get('label'))

    y = torch.tensor(y)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)

    x = TensorDataset(input_ids, attention_mask, token_type_ids)
    return SequenceClassificationDataset(x, y)


data_dir = '../datas/datas/'
train_dir = os.path.join(data_dir,'train.json')
val_dir = os.path.join(data_dir,'val.json')
test_dir = os.path.join(data_dir, 'test.json')

labels = ['0','1','2','3']

# dataset
save_data_dir = os.path.join(data_dir, 'processed')

save_dir = './output'
checkpoint_dir = os.path.join(save_dir, 'checkpoints')
model_save_dir = os.path.join(save_dir, 'saved')
tblog_dir = os.path.join(save_dir, 'runs')

os.makedirs(save_data_dir,exist_ok = True)
os.makedirs(save_dir,exist_ok = True)

train_datas = utils.read_json(train_dir)
trainset = transfer_datas(train_datas)

val_datas = utils.read_json(val_dir)
valset = transfer_datas(val_datas)

test_datas = utils.read_json(test_dir)
testset = transfer_datas(test_datas)

print('type trainset: ', type(trainset))
print(trainset[0])

num_labels = 4
model = BertSequenceClassification(bert_model_dir, num_labels=num_labels)

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

runner = Trainer(model=model, trainset=trainset,
                         valset=valset, train_batch_size=batch_size,
                         checkpoint_dir=checkpoint_dir, save_dir=model_save_dir,
                         tblog_dir=tblog_dir, tblog_freq=100,
                         optimizer=optimizer, lr_scheduler=lr_scheduler,
                         checkpoint_freq = 500,
                         val_batch_size=batch_size, max_epochs=50, device='2')

# runner.run()

model_dir='./output/saved/'
infer = antiHarassment_api.Inferer(model_dir, tokenizer_dir, device = '2')
pred, proba = infer.infer(test_datas)
true = [i.get('label') for i in test_datas]
print('classification report')
print(classification_report(true, pred))

def get_wrong_datas(true, pred, datas):
    id_ = []
    speaker = []
    dialogue = []
    w_true = []
    w_pred = []
    for idx, i in enumerate(datas):
        if true[idx] != pred[idx]:
            id_.append(i.get('_id'))
            speaker.append(i.get('speaker'))
            dialogue.append(i.get('text_a'))
            w_true.append(true[idx])
            w_pred.append(pred[idx])

    df_wrong = pd.DataFrame(data={'_id': id_, 
                                  'speaker': speaker,
                                  'dialogue': dialogue, 
                                  'true': w_true, 
                                  'pred': w_pred})
    return df_wrong

outp = './output/wrong/wrong.xlsx'
df_wrong = get_wrong_datas(true, pred, test_datas)
df_wrong.to_excel(outp, index=False)
