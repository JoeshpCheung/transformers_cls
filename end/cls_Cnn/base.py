#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data.dataloader import default_collate

from aleph.data.textclassification.bert import Processor
from aleph.metrics.custom import Accuracy
from aleph.trainer import TrainRunnerBase
from ignite.metrics import Recall,Fbeta, Precision


class MultiLabelProcessor(Processor):
    def label2index(self, label):
        return label

    def index2label(self, index):
        return index

'''
class Trainer(TrainRunnerBase):
    def __init__(self, threshold=0.5, **kwargs):
        print('kwargs: ', kwargs)
        kwargs['metric_name'] = kwargs.get('metric_name', 'f')
        kwargs['criterion'] = kwargs.get('criterion', nn.BCEWithLogitsLoss())
        self.threshold = threshold
        super(Trainer, self).__init__(**kwargs)

    def calc_loss(self, batch):
        x, labels = batch
        # print('x: ', x.shape, 'labels: ', labels.shape)
        logits = self.model(x)
        # logits = torch.softmax(logits, 1)
        logits = logits.sigmoid()
        print('logits: ', logits.shape, labels.shape)
        print(logits, labels)
        print('logits: ', type(logits), type(labels))
        
        loss = self.criterion(logits, labels)
        print('loss: ', loss)
        return loss

    def evaluate(self, batch):
        x, labels = batch
        logits = self.model(x)
        loss = self.criterion(logits, labels)
        loss = loss.item()
        logits = logits.sigmoid()
        y_pred = (logits > self.threshold).type(torch.long).tolist()
        labels = labels.type(torch.long).tolist()
        return y_pred, labels, loss

    def output_transform(self, output):
        y_pred, y, _ = output
        print('transform: ', y_pred, y)
        print('transform: ', type(y_pred), type(y))
        y_pred = torch.tensor(y_pred).to(self.device)
        y = torch.tensor(y).to(self.device)
        return y_pred, y

    def start(self):
        Accuracy(output_transform = self.output_transform).attach(self.evaluator, 'acc')
        Recall(output_transform = self.output_transform).attach(self.evaluator, 'r')
        Precision(output_transform = self.output_transform).attach(self.evaluator, 'p')
        Fbeta(beta = 1.0, output_transform = self.output_transform).attach(self.evaluator, 'f')
        self.logger.info('START TRAINING')

    def dataset2dataloader(self, dataset, batch_size, **kwargs):

        is_train = dataset.y is not None

        def collate_fn(batch):
            if is_train:
                x, y = default_collate(batch)
                # x = [_x.to(self.device) for _x in x]
                x = torch.tensor(x, dtype=torch.float).to(self.device)
                y = y.type(torch.float)
                y = y.to(self.device)
                return x, y
            else:
                x = default_collate(batch)
                x = torch.tensor(x, dtype=torch.float).to(self.device)
                # x = [_x.to(self.device) for _x in x]
                return x

        kwargs.update(collate_fn=collate_fn, pin_memory=False)
        return super(Trainer, self).dataset2dataloader(
            dataset, batch_size, **kwargs)


'''

class Trainer(TrainRunnerBase):
    def __init__(self, threshold=0.5, **kwargs):
        print('kwargs: ', kwargs)
        kwargs['metric_name'] = kwargs.get('metric_name', 'f')
        kwargs['criterion'] = kwargs.get('criterion', nn.CrossEntropyLoss())
        self.threshold = threshold
        super(Trainer, self).__init__(**kwargs)

    def calc_loss(self, batch):
        x, labels = batch
        # print('x: ', x.shape, 'labels: ', labels.shape)
        logits = self.model(x)
        logits = torch.softmax(logits, 1)
        # print('logits: ', logits.shape, labels.shape)
        # print(logits, labels)
        # print('logits: ', type(logits), type(labels))
        loss = self.criterion(logits, labels.long())
        # print('loss: ', loss)
        return loss

    def evaluate(self, batch):
        x, labels = batch
        logits = self.model(x)
        # print('logits: ', logits, 'labels: ', labels)
        logits = torch.softmax(logits, 1)
        # print('evaluate', logits.shape, labels.shape)
        loss = self.criterion(logits, labels.long())
        loss = loss.item()

        # logits = logits.sigmoid()
        
        y_pred = []
        tmp_max = logits.max(1).values.tolist()
        for idx, i in enumerate(logits):
            tmp = (logits[idx] == tmp_max[idx]).type(torch.long).tolist()
            y_pred.append(tmp)
        
        # y_pred = logits
        # y_pred = (logits > self.threshold).type(torch.long).tolist()
        
        labels = labels.type(torch.long).tolist()
        y = []
        for idx, i in enumerate(labels):
            tmp = [0 for a in range(11)]  # 11->num class
            tmp[i] = 1
            y.append(tmp)

        # print('evaluate: ', y_pred, labels)
        return y_pred, y, loss

    def output_transform(self, output):
        y_pred, y, _ = output
        # print('transform: ', y_pred, y)
        # print('transform: ', type(y_pred), type(y))
        y_pred = torch.tensor(y_pred).to(self.device)
        y = torch.tensor(y).to(self.device)
        return y_pred, y

    def start(self):
        Accuracy(output_transform = self.output_transform).attach(self.evaluator, 'acc')
        Recall(output_transform = self.output_transform).attach(self.evaluator, 'r')
        Precision(output_transform = self.output_transform).attach(self.evaluator, 'p')
        Fbeta(beta = 1.0, output_transform = self.output_transform).attach(self.evaluator, 'f')
        self.logger.info('START TRAINING')

    def dataset2dataloader(self, dataset, batch_size, **kwargs):
        is_train = dataset.y is not None

        def collate_fn(batch):
            if is_train:
                x, y = default_collate(batch)
                x = torch.tensor(x, dtype=torch.float).to(self.device)
                y = y.type(torch.float)
                y = y.to(self.device)
                return x, y
            else:
                x = default_collate(batch)
                x = torch.tensor(x, dtype=torch.float).to(self.device)
                return x

        kwargs.update(collate_fn=collate_fn, pin_memory=False)
        return super(Trainer, self).dataset2dataloader(
            dataset, batch_size, **kwargs)
# '''
