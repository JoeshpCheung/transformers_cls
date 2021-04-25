#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    ~~~~~~~~~~~~~~~~~~~~~~~
    :author: JoeshpCheung
    :python version: 3.6
"""
import torch.nn as nn
import torch
import torch.nn.functional as F

class Cnn(nn.Module):
    def __init__(self):
        super(Cnn,self).__init__()
        self.conv1 = nn.Conv2d(20, 5, kernel_size=1,padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16,120,kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120,84)
        self.fc2 = nn.Linear(84,10)
        self.logsoftmax = nn.LogSoftmax()

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self,x):
        in_size = x.size(0)
        batch, h, l = x.shape
        # x = torch.reshape(x, [batch, h, l, 1])
        x = x.unsqueeze(1)

        print('x')
        print(type(x))
        print(x)
        out = self.relu(self.mp(self.conv1(x)))

        out = self.relu(self.mp(self.conv2(out)))
        out = self.relu(self.conv3(out))
        out = out.view(in_size, -1)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return self.logsoftmax(out)


class TextCnn(nn.Module):
    def __init__(self, num_classes, dropout_prob=0.3, num_filters=256, filter_sizes=(2, 3, 5, 8, 13, 21, 34, 55, 87)):
    # def __init__(self, num_classes, dropout_prob=0.3, num_filters=256, filter_sizes=(2, 3, 4)):
        super(TextCnn, self).__init__()

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (k, 20)) for k in filter_sizes])
        # self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
        # self.relu = nn.ReLU()

    def conv_and_pool(self, x, conv):
        # print('x1: ', x.shape)
        x = F.relu(conv(x))
        # print('x2: ', x.shape)
        x = x.squeeze(3)
        # print('hallo')
        # print('x3: ', x.shape)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # x = torch.tensor(x, dtype=torch.float).to(device)
        
        x = x.permute(0,2,1)

        out = x.unsqueeze(1)
        # print('out1.shape: ', out.shape)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        # print('out2.shape: ', out.shape)
        # out = self.relu(self.conv2(out))
        
        out = self.dropout(out)
        out = self.fc(out)
        return out
