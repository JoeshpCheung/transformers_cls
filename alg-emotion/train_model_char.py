#!/usr/bin/env python
# coding=utf-8

"""
    train_model.py
    ~~~~~~~~~~~~~~~~~~~~~~~

    Description of this file

    :author: Tianxiang Zhang
    :python version: 3.6

"""
import os
import sys
import json
import jieba
import pickle
import joblib
import logging
# format='%(asctime)s: %(message)s', datefmt='%Y-%m-%d: %H:%M '
logging.basicConfig(filename='log/train_model_char.log', filemode='a+', level=logging.INFO, format='%(message)s')

#import jason_utils
import numpy as np
import pandas as pd
import importlib
importlib.reload(sys)
import keras
from keras import backend as K
import matplotlib.pyplot as plt
from collections import defaultdict
from keras.engine.topology import Layer
from keras.preprocessing import sequence
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
base_path = os.path.dirname(os.path.abspath(__file__))
os.environ["PATH"] += os.pathsep + '/home/tungee/anaconda3/lib/python3.5/site-packages'

char2idx = json.load(open('data/char2idx.json'))
iteration = 50
EPOCHS = 50
EMBEDDING_DIM = 128
MAX_LEN = 200
base_path = os.path.dirname(os.path.abspath(__file__))

def load_data(path):
    """
    载入数据
    """
    df = pd.read_csv(path, sep='\t', header=None)
    tmp = list(df[2])
    tmp = ['U' if pd.isnull(i) else i for i in tmp]
    x_data = [[char2idx[i] if i in char2idx else char2idx['U'] for i in j] for j in tmp]
    y_data = list(df[1].replace('正面', 0).replace('负面', 1).replace('中性', 2))

    return x_data, y_data

def data_transform(x_data, used_fields): # concatenate all inputs to 1
    """
    对输入数据的不同字段进行转换：padding或者one-hot编码
    """
    x_inputs = []
    for i, field in enumerate(used_fields):
        #for j in range(len(x_data[field])):
        x_inputs.append(
            sequence.pad_sequences(x_data[field], maxlen=field_lens[field], padding='post', truncating='post')
        )
    record = []
    for i in range(len(x_inputs[0])):
        tmp = []
        for j in range(len(x_inputs)):
            tmp += list(x_inputs[j][i])
        record.append(tmp)
    record = [np.array(i) for i in record]
    record = np.array(record)
    return record

class Position_Embedding(Layer):
    def __init__(self, size=None, mode='sum', **kwargs):
        self.size = size  # 必须为偶数
        self.mode = mode
        super(Position_Embedding, self).__init__(**kwargs)

    def call(self, x):
        if (self.size == None) or (self.mode == 'sum'):
            self.size = int(x.shape[-1])
        batch_size, seq_len = K.shape(x)[0], K.shape(x)[1]
        position_j = 1. / K.pow(10000., 2 * K.arange(self.size / 2, dtype='float32' \
                                             ) / self.size)
        position_j = K.expand_dims(position_j, 0)
        position_i = K.cumsum(K.ones_like(x[:, :, 0]), 1) - 1  # K.arange不支持变长，只好用这种方法生成
        position_i = K.expand_dims(position_i, 2)
        position_ij = K.dot(position_i, position_j)
        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)
        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return K.concatenate([position_ij, x], 2)

    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2] + self.size)

class Attention(Layer):

    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head*size_per_head
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:,0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape)-2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def call(self, x):
        #如果只传入Q_seq,K_seq,V_seq，那么就不做Mask
        #如果同时传入Q_seq,K_seq,V_seq,Q_len,V_len，那么对多余部分做Mask
        if len(x) == 3:
            Q_seq,K_seq,V_seq = x
            Q_len,V_len = None,None
        elif len(x) == 5:
            Q_seq,K_seq,V_seq,Q_len,V_len = x
        #对Q、K、V做线性变换
        print('Q_seqf0', Q_seq)
        Q_seq = K.dot(Q_seq, self.WQ)
        print('WQ', self.WQ)
        print('Q_seqf', Q_seq)
        print('K.shape(Q_seq)', K.shape(Q_seq))
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        print('Q_seqf1', Q_seq)
        Q_seq = K.permute_dimensions(Q_seq, (0, 2, 1, 3))
        print('Q_seq', Q_seq)
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0, 2, 1, 3))
        print('K_seq', K_seq)
        V_seq = K.dot(V_seq, self.WV)
        print('WV', self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        print('y_seq', V_seq)
        V_seq = K.permute_dimensions(V_seq, (0,2,1,3))
        print('V_seq1', V_seq)
        #计算内积，然后mask，然后softmax
        #A = K.batch_dot(Q_seq, K_seq, axes=[3,3]) / self.size_per_head**0.5
        A = K.batch_dot(Q_seq, K_seq, axes=[3, 3])
        B = K.dot(Q_seq, K_seq)
        print('B', B)
        print('Q', Q_seq, 'K_seq', K_seq)
        print('A0', A)
        A = A / self.size_per_head**0.5
        print('A1', A)
        A = K.permute_dimensions(A, (0,3,2,1))
        print('A2', A)
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0,3,2,1))
        A = K.softmax(A)
        #输出并mask
        O_seq = K.batch_dot(A, V_seq, axes=[3,2])
        O_seq = K.permute_dimensions(O_seq, (0,2,1,3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)

class Self_Attention(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Self_Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        # inputs.shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3, input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(Self_Attention, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])
        print("WQ.shape", WQ.shape)
        print("K.permute_dimensions(WK, [0, 2, 1]).shape", K.permute_dimensions(WK, [0, 2, 1]).shape)
        QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))
        dk = int(WQ.shape[-1])
        QK = QK / (dk ** 0.5)
        QK = K.softmax(QK)
        print("QK.shape", QK.shape)
        print('WV.shape', WV.shape)
        V = K.batch_dot(QK, WV)
        print('V.shape', V.shape)
        return V

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

class nlpModel(object):
    def __init__(self, lens, plot=True):
        self.lens = lens
        self.plot = plot

    def Model1(self):# fast_text
        inputs = keras.Input(shape=(self.lens, ), )
        embeddings = keras.layers.Embedding(input_dim=len(char2idx), output_dim=EMBEDDING_DIM)(inputs)
        pooling_layer = keras.layers.GlobalAveragePooling1D()(embeddings)
        dropout_layer = keras.layers.Dropout(0.3)(pooling_layer)
        y = keras.layers.Dense(1, activation='sigmoid')(dropout_layer)
        model = keras.Model(inputs=inputs, outputs=[y])
        if self.plot:
            keras.utils.plot_model(model, to_file='model/Model1.png', show_shapes=True)
        return model

    def Model2(self):# Global average&max&std pooling + dense
        inputs = keras.Input(shape=(self.lens, ), )
        embed = keras.layers.Embedding(input_dim=len(char2idx), output_dim=EMBEDDING_DIM)(inputs)
        pooling_layer = keras.layers.concatenate([
            keras.layers.GlobalAveragePooling1D()(embed),
            keras.layers.GlobalMaxPooling1D()(embed),
            keras.layers.Lambda(lambda x: keras.backend.std(x, axis=1), name=('global_std_pooling1d_'))(embed)
        ])
        y = keras.layers.Dense(1, activation='sigmoid')(pooling_layer)
        model = keras.Model(inputs=inputs, outputs=[y])
        if self.plot:
            keras.utils.plot_model(model, to_file='model/Model2.png', show_shapes=True)
        return model

    def Model3(self):# embed + attention + Global avgpooling + dense
        inputs = keras.Input(shape=(self.lens, ), )
        embeddings = keras.layers.Embedding(input_dim=len(char2idx), output_dim=EMBEDDING_DIM)(inputs)
        attention_layer = Self_Attention(EMBEDDING_DIM)(embeddings)
        pooling_layer = keras.layers.GlobalAveragePooling1D()(attention_layer)
        dropout_layer = keras.layers.Dropout(0.3)(pooling_layer)
        y = keras.layers.Dense(1, activation='sigmoid')(dropout_layer)
        model = keras.Model(inputs=inputs, outputs=[y])
        if self.plot:
            keras.utils.plot_model(model, to_file='model/Model3.png', show_shapes=True)
        return model

    def Model4(self):# CNN best for ToB
        inputs = keras.Input(shape=(self.lens, ), )
        embed = keras.layers.Embedding(input_dim=len(char2idx), output_dim=EMBEDDING_DIM)(inputs)
        cnn_layer = keras.layers.Conv1D(256, 5, padding='same', strides=1, activation='relu')(embed)
        pooling_layer = keras.layers.MaxPool1D(pool_size=46)(cnn_layer)
        flatten_layer = keras.layers.Flatten()(pooling_layer)
        dropout = keras.layers.Dropout(0.3)(flatten_layer)
        y = keras.layers.Dense(1, activation='sigmoid')(dropout)
        model = keras.Model(inputs=inputs, outputs=[y])
        if self.plot:
            keras.utils.plot_model(model, to_file='model/Model4.png', show_shapes=True)
        print(model.summary())
        return model

    def Model4_softmax(self):# CNN best for ToB
        inputs = keras.Input(shape=(self.lens, ), )
        embed = keras.layers.Embedding(input_dim=len(char2idx), output_dim=EMBEDDING_DIM)(inputs)
        cnn_layer = keras.layers.Conv1D(256, 5, padding='same', strides=1, activation='relu')(embed)
        pooling_layer = keras.layers.MaxPool1D(pool_size=46)(cnn_layer)
        flatten_layer = keras.layers.Flatten()(pooling_layer)
        dropout = keras.layers.Dropout(0.3)(flatten_layer)
        y = keras.layers.Dense(1, activation='softmax')(dropout)
        model = keras.Model(inputs=inputs, outputs=[y])
        if self.plot:
            keras.utils.plot_model(model, to_file='model/Model4.png', show_shapes=True)
        print(model.summary())
        return model

    def Model4_1(self):# CNN
        inputs = keras.Input(shape=(self.lens, ), )
        mask = keras.layers.Masking(mask_value=0)(inputs)
        embed = keras.layers.Embedding(input_dim=len(char2idx), output_dim=EMBEDDING_DIM)(mask)
        cnn_layer = keras.layers.Conv1D(256, 5, padding='same', strides=1, activation='relu')(embed)
        pooling_layer = keras.layers.MaxPool1D(pool_size=46)(cnn_layer)
        flatten_layer = keras.layers.Flatten()(pooling_layer)
        dropout = keras.layers.Dropout(0.3)(flatten_layer)
        y = keras.layers.Dense(1, activation='sigmoid')(dropout)
        model = keras.Model(inputs=inputs, outputs=[y])
        if self.plot:
            keras.utils.plot_model(model, to_file='model/Model4.png', show_shapes=True)
        print(model.summary())
        return model

    def Model5(self):# ATT-CNN
        inputs = keras.Input(shape=(self.lens, ), )
        embed = keras.layers.Embedding(input_dim=len(char2idx), output_dim=EMBEDDING_DIM)(inputs)
        attention_layer = Self_Attention(EMBEDDING_DIM)(embed)
        #attention_layer = Attention(8, 16)([embed, embed, embed])
        cnn_layer = keras.layers.Conv1D(256, 5, padding='same', strides=1, activation='relu')(attention_layer)
        pooling_layer = keras.layers.MaxPool1D(pool_size=46)(cnn_layer)
        flatten_layer = keras.layers.Flatten()(pooling_layer)
        dropout = keras.layers.Dropout(0.3)(flatten_layer)
        y = keras.layers.Dense(1, activation='sigmoid')(dropout)
        model = keras.Model(inputs=inputs, outputs=[y])
        if self.plot:
            keras.utils.plot_model(model, to_file='model/Model5.png', show_shapes=True)
        print(model.summary())
        return model

    def Model5_1(self):# ATT-CNN
        inputs = keras.Input(shape=(self.lens, ), )
        embed = keras.layers.Embedding(input_dim=len(char2idx), output_dim=EMBEDDING_DIM)(inputs)
        cnn_layer = keras.layers.Conv1D(256, 5, padding='same', strides=1, activation='relu')(embed)
        attention_layer = Self_Attention(EMBEDDING_DIM)(cnn_layer)
        pooling_layer = keras.layers.MaxPool1D(pool_size=46)(attention_layer)
        flatten_layer = keras.layers.Flatten()(pooling_layer)
        dropout = keras.layers.Dropout(0.3)(flatten_layer)
        y = keras.layers.Dense(1, activation='sigmoid')(dropout)
        model = keras.Model(inputs=inputs, outputs=[y])
        if self.plot:
            keras.utils.plot_model(model, to_file='model/Model5_1.png', show_shapes=True)
        print(model.summary())
        return model

    def Model5_2(self):# ATT-CNN
        inputs = keras.Input(shape=(self.lens, ), )
        embed = keras.layers.Embedding(input_dim=len(char2idx), output_dim=EMBEDDING_DIM)(inputs)
        attention_layer = SeqSelfAttention(attention_activation='sigmoid')(embed)
        cnn_layer = keras.layers.Conv1D(256, 5, padding='same', strides=1, activation='relu')(attention_layer)
        pooling_layer = keras.layers.MaxPool1D(pool_size=46)(cnn_layer)
        flatten_layer = keras.layers.Flatten()(pooling_layer)
        dropout = keras.layers.Dropout(0.3)(flatten_layer)
        y = keras.layers.Dense(1, activation='sigmoid')(dropout)
        model = keras.Model(inputs=inputs, outputs=[y])
        if self.plot:
            keras.utils.plot_model(model, to_file='model/Model5_2.png', show_shapes=True)
        print(model.summary())
        return model

    def Model5_3(self):# ATT-CNN
        inputs = keras.Input(shape=(self.lens, ), )
        embed = keras.layers.Embedding(input_dim=len(char2idx), output_dim=EMBEDDING_DIM)(inputs)
        cnn_layer = keras.layers.Conv1D(256, 5, padding='same', strides=1, activation='relu')(embed)
        attention_layer = SeqSelfAttention(attention_activation='sigmoid')(cnn_layer)
        pooling_layer = keras.layers.MaxPool1D(pool_size=46)(attention_layer)
        flatten_layer = keras.layers.Flatten()(pooling_layer)
        dropout = keras.layers.Dropout(0.3)(flatten_layer)
        y = keras.layers.Dense(1, activation='sigmoid')(dropout)
        model = keras.Model(inputs=inputs, outputs=[y])
        if self.plot:
            keras.utils.plot_model(model, to_file='model/Model5_3.png', show_shapes=True)
        print(model.summary())
        return model

    def Model5_4(self):# ATT-CNN
        inputs = keras.Input(shape=(self.lens, ), )
        embed = keras.layers.Embedding(input_dim=len(char2idx), output_dim=EMBEDDING_DIM)(inputs)
        embed = Position_Embedding()(embed)
        attention_layer = Self_Attention(EMBEDDING_DIM)(embed)
        cnn_layer = keras.layers.Conv1D(256, 5, padding='same', strides=1, activation='relu')(attention_layer)
        pooling_layer = keras.layers.GlobalAveragePooling1D()(cnn_layer)
        dropout = keras.layers.Dropout(0.5)(pooling_layer)
        y = keras.layers.Dense(1, activation='sigmoid')(dropout)
        model = keras.Model(inputs=inputs, outputs=[y])
        if self.plot:
            keras.utils.plot_model(model, to_file='model/Model5_4.png', show_shapes=True)
        print(model.summary())
        return model

    def Model5_5(self):# Transformer - CNN - pooling - flatten
        inputs = keras.Input(shape=(self.lens, ), )
        embed = keras.layers.Embedding(input_dim=len(char2idx), output_dim=EMBEDDING_DIM)(inputs)
        embed = Position_Embedding()(embed)
        attention_layer = Self_Attention(EMBEDDING_DIM)(embed)
        cnn_layer = keras.layers.Conv1D(256, 5, padding='same', strides=1, activation='relu')(attention_layer)
        pooling_layer = keras.layers.concatenate([
            keras.layers.GlobalAveragePooling1D()(cnn_layer),
            keras.layers.GlobalMaxPooling1D()(cnn_layer),
            keras.layers.Lambda(lambda x: keras.backend.std(x, axis=1), name=('global_std_pooling1d_'))(cnn_layer)
        ])
        dropout = keras.layers.Dropout(0.5)(pooling_layer)
        y = keras.layers.Dense(1, activation='sigmoid')(dropout)
        model = keras.Model(inputs=inputs, outputs=[y])
        if self.plot:
            keras.utils.plot_model(model, to_file='model/Model5.png', show_shapes=True)
        print(model.summary())
        return model

    def Model6(self):# combine input - textcnn - flatten - dense7 - dense1
        inputs = keras.Input(shape=(self.lens, ), )
        embed = keras.layers.Embedding(input_dim=len(char2idx), output_dim=EMBEDDING_DIM)(inputs)

        cnn1 = keras.layers.Conv1D(256, 3, padding='same', strides=1, activation='relu')(embed)
        cnn1 = keras.layers.MaxPool1D(pool_size=48)(cnn1)

        cnn2 = keras.layers.Conv1D(256, 4, padding='same', strides=1, activation='relu')(embed)
        cnn2 = keras.layers.MaxPool1D(pool_size=47)(cnn2)

        cnn3 = keras.layers.Conv1D(256, 5, padding='same', strides=1, activation='relu')(embed)
        cnn3 = keras.layers.MaxPool1D(pool_size=46)(cnn3)

        cnn = keras.layers.concatenate([cnn1, cnn2, cnn3], axis=-1)
        flat = keras.layers.Flatten()(cnn)
        drop = keras.layers.Dropout(0.3)(flat)
        dense = keras.layers.Dense(7, activation='relu')(drop)
        y = keras.layers.Dense(1, activation='sigmoid')(dense)
        model = keras.Model(inputs=inputs, outputs=[y])
        if self.plot:
            keras.utils.plot_model(model, to_file='model/Model6.png', show_shapes=True)
        print(model.summary())
        return model

    def Model7(self):# combine input - textcnn - flatten - dense1 # best for ToB
        inputs = keras.Input(shape=(self.lens, ), )
        embed = keras.layers.Embedding(input_dim=len(char2idx), output_dim=EMBEDDING_DIM)(inputs)

        cnn1 = keras.layers.Conv1D(256, 3, padding='same', strides=1, activation='relu')(embed)
        cnn1 = keras.layers.MaxPool1D(pool_size=48)(cnn1)

        cnn2 = keras.layers.Conv1D(256, 4, padding='same', strides=1, activation='relu')(embed)
        cnn2 = keras.layers.MaxPool1D(pool_size=47)(cnn2)

        cnn3 = keras.layers.Conv1D(256, 5, padding='same', strides=1, activation='relu')(embed)
        cnn3 = keras.layers.MaxPool1D(pool_size=46)(cnn3)

        cnn = keras.layers.concatenate([cnn1, cnn2, cnn3], axis=-1)
        flat = keras.layers.Flatten()(cnn)
        drop = keras.layers.Dropout(0.5)(flat)
        y = keras.layers.Dense(1, activation='sigmoid')(drop)
        model = keras.Model(inputs=inputs, outputs=[y])
        if self.plot:
            keras.utils.plot_model(model, to_file='model/Model7.png', show_shapes=True)
        print(model.summary())
        return model

    def Model7_1(self):# combine input - textcnn - flatten - dense1 #best for ToC
        inputs = keras.Input(shape=(self.lens, ), )
        mask = keras.layers.Masking(mask_value=0)(inputs)
        embed = keras.layers.Embedding(input_dim=len(char2idx), output_dim=EMBEDDING_DIM)(mask)

        cnn1 = keras.layers.Conv1D(256, 3, padding='same', strides=1, activation='relu')(embed)
        cnn1 = keras.layers.MaxPool1D(pool_size=48)(cnn1)

        cnn2 = keras.layers.Conv1D(256, 4, padding='same', strides=1, activation='relu')(embed)
        cnn2 = keras.layers.MaxPool1D(pool_size=47)(cnn2)

        cnn3 = keras.layers.Conv1D(256, 5, padding='same', strides=1, activation='relu')(embed)
        cnn3 = keras.layers.MaxPool1D(pool_size=46)(cnn3)

        cnn = keras.layers.concatenate([cnn1, cnn2, cnn3], axis=-1)
        flat = keras.layers.Flatten()(cnn)
        drop = keras.layers.Dropout(0.5)(flat)
        y = keras.layers.Dense(1, activation='sigmoid')(drop)
        model = keras.Model(inputs=inputs, outputs=[y])
        if self.plot:
            keras.utils.plot_model(model, to_file='model/Model7.png', show_shapes=True)
        print(model.summary())
        return model

    def Model7_2(self):# combine input - attention - textcnn - flatten - dense1
        inputs = keras.Input(shape=(self.lens, ), )
        #mask = keras.layers.Masking(mask_value=0)(inputs)
        embed = keras.layers.Embedding(input_dim=len(char2idx), output_dim=EMBEDDING_DIM)(inputs)
        attention_layer = Self_Attention(EMBEDDING_DIM)(embed)

        cnn1 = keras.layers.Conv1D(256, 3, padding='same', strides=1, activation='relu')(attention_layer)
        cnn1 = keras.layers.MaxPool1D(pool_size=48)(cnn1)

        cnn2 = keras.layers.Conv1D(256, 4, padding='same', strides=1, activation='relu')(attention_layer)
        cnn2 = keras.layers.MaxPool1D(pool_size=47)(cnn2)

        cnn3 = keras.layers.Conv1D(256, 5, padding='same', strides=1, activation='relu')(attention_layer)
        cnn3 = keras.layers.MaxPool1D(pool_size=46)(cnn3)

        cnn = keras.layers.concatenate([cnn1, cnn2, cnn3], axis=-1)
        flat = keras.layers.Flatten()(cnn)
        drop = keras.layers.Dropout(0.5)(flat)
        y = keras.layers.Dense(1, activation='sigmoid')(drop)
        model = keras.Model(inputs=inputs, outputs=[y])
        if self.plot:
            keras.utils.plot_model(model, to_file='model/Model7.png', show_shapes=True)
        print(model.summary())
        return model

    def Model7_3(self):# combine input - attention - textcnn - flatten - dense1
        inputs = keras.Input(shape=(self.lens, ), )
        mask = keras.layers.Masking(mask_value=0)(inputs)
        embed = keras.layers.Embedding(input_dim=len(char2idx), output_dim=EMBEDDING_DIM)(mask)
        attention_layer = Self_Attention(EMBEDDING_DIM)(embed)

        cnn1 = keras.layers.Conv1D(256, 3, padding='same', strides=1, activation='relu')(attention_layer)
        cnn1 = keras.layers.MaxPool1D(pool_size=48)(cnn1)

        cnn2 = keras.layers.Conv1D(256, 4, padding='same', strides=1, activation='relu')(attention_layer)
        cnn2 = keras.layers.MaxPool1D(pool_size=47)(cnn2)

        cnn3 = keras.layers.Conv1D(256, 5, padding='same', strides=1, activation='relu')(attention_layer)
        cnn3 = keras.layers.MaxPool1D(pool_size=46)(cnn3)

        cnn = keras.layers.concatenate([cnn1, cnn2, cnn3], axis=-1)
        flat = keras.layers.Flatten()(cnn)
        drop = keras.layers.Dropout(0.5)(flat)
        y = keras.layers.Dense(1, activation='sigmoid')(drop)
        model = keras.Model(inputs=inputs, outputs=[y])
        if self.plot:
            keras.utils.plot_model(model, to_file='model/Model7.png', show_shapes=True)
        print(model.summary())
        return model

    def Model7_4(self):# combine input - embed - pos_embed - attention - textcnn - flatten - dense1
        inputs = keras.Input(shape=(self.lens, ), )
        #mask = keras.layers.Masking(mask_value=0)(inputs)
        embed = keras.layers.Embedding(input_dim=len(char2idx), output_dim=EMBEDDING_DIM)(inputs)
        embed = Position_Embedding()(embed)
        attention_layer = Self_Attention(EMBEDDING_DIM)(embed)

        cnn1 = keras.layers.Conv1D(256, 3, padding='same', strides=1, activation='relu')(attention_layer)
        cnn1 = keras.layers.MaxPool1D(pool_size=48)(cnn1)

        cnn2 = keras.layers.Conv1D(256, 4, padding='same', strides=1, activation='relu')(attention_layer)
        cnn2 = keras.layers.MaxPool1D(pool_size=47)(cnn2)

        cnn3 = keras.layers.Conv1D(256, 5, padding='same', strides=1, activation='relu')(attention_layer)
        cnn3 = keras.layers.MaxPool1D(pool_size=46)(cnn3)

        cnn = keras.layers.concatenate([cnn1, cnn2, cnn3], axis=-1)
        flat = keras.layers.Flatten()(cnn)
        drop = keras.layers.Dropout(0.5)(flat)
        y = keras.layers.Dense(1, activation='sigmoid')(drop)
        model = keras.Model(inputs=inputs, outputs=[y])
        if self.plot:
            keras.utils.plot_model(model, to_file='model/Model7.png', show_shapes=True)
        print(model.summary())
        return model

    def Model7_5(self):# combine input - embed - pos_embed - attention - textcnn - flatten - dense1
        inputs = keras.Input(shape=(self.lens, ), )
        mask = keras.layers.Masking(mask_value=0)(inputs)
        embed = keras.layers.Embedding(input_dim=len(char2idx), output_dim=EMBEDDING_DIM)(mask)
        embed = Position_Embedding()(embed)
        attention_layer = Self_Attention(EMBEDDING_DIM)(embed)

        cnn1 = keras.layers.Conv1D(256, 3, padding='same', strides=1, activation='relu')(attention_layer)
        cnn1 = keras.layers.MaxPool1D(pool_size=48)(cnn1)

        cnn2 = keras.layers.Conv1D(256, 4, padding='same', strides=1, activation='relu')(attention_layer)
        cnn2 = keras.layers.MaxPool1D(pool_size=47)(cnn2)

        cnn3 = keras.layers.Conv1D(256, 5, padding='same', strides=1, activation='relu')(attention_layer)
        cnn3 = keras.layers.MaxPool1D(pool_size=46)(cnn3)

        cnn = keras.layers.concatenate([cnn1, cnn2, cnn3], axis=-1)
        flat = keras.layers.Flatten()(cnn)
        drop = keras.layers.Dropout(0.5)(flat)
        y = keras.layers.Dense(1, activation='sigmoid')(drop)
        model = keras.Model(inputs=inputs, outputs=[y])
        if self.plot:
            keras.utils.plot_model(model, to_file='model/Model7.png', show_shapes=True)
        print(model.summary())
        return model

    def Model8(self):# Bilstm step + flatten + dense
        inputs = keras.Input(shape=(self.lens, ), )
        embed = keras.layers.Embedding(input_dim=len(char2idx), output_dim=EMBEDDING_DIM)(inputs)
        rnn_layer = keras.layers.Bidirectional(keras.layers.CuDNNLSTM(EMBEDDING_DIM, return_sequences=True))(embed)
        flatten_layer = keras.layers.Flatten()(rnn_layer)
        dropout = keras.layers.Dropout(0.3)(flatten_layer)
        y = keras.layers.Dense(1, activation='sigmoid')(dropout)
        model = keras.Model(inputs=inputs, outputs=[y])
        if self.plot:
            keras.utils.plot_model(model, to_file='model/Model8.png', show_shapes=True)
        print(model.summary())
        return model

    def Model9(self):# Bilstm
        inputs = keras.Input(shape=(self.lens, ), )
        embed = keras.layers.Embedding(input_dim=len(char2idx), output_dim=EMBEDDING_DIM)(inputs)
        rnn_layer = keras.layers.Bidirectional(keras.layers.CuDNNLSTM(EMBEDDING_DIM))(embed)
        dropout = keras.layers.Dropout(0.3)(rnn_layer)
        y = keras.layers.Dense(1, activation='sigmoid')(dropout)
        model = keras.Model(inputs=inputs, outputs=[y])
        if self.plot:
            keras.utils.plot_model(model, to_file='model/Model9.png', show_shapes=True)
        print(model.summary())
        return model

    def Model11(self):# embed + mask + Bilstm
        inputs = keras.Input(shape=(self.lens, ), )
        mask = keras.layers.Masking(mask_value=0)(inputs)
        embed = keras.layers.Embedding(input_dim=len(char2idx), output_dim=EMBEDDING_DIM)(mask)
        rnn_layer = keras.layers.Bidirectional(keras.layers.CuDNNLSTM(EMBEDDING_DIM))(embed)
        dropout = keras.layers.Dropout(0.3)(rnn_layer)
        y = keras.layers.Dense(1, activation='sigmoid')(dropout)
        model = keras.Model(inputs=inputs, outputs=[y])
        if self.plot:
            keras.utils.plot_model(model, to_file='model/Model11.png', show_shapes=True)
        print(model.summary())
        return model

    def Model10(self):# Bilstm step + attention + GobalMaxPooling + dense
        inputs = keras.Input(shape=(self.lens, ), )
        embed = keras.layers.Embedding(input_dim=len(char2idx), output_dim=EMBEDDING_DIM)(inputs)
        rnn_layer = keras.layers.Bidirectional(keras.layers.CuDNNLSTM(EMBEDDING_DIM, return_sequences=True))(embed)
        attention_layer = Self_Attention(EMBEDDING_DIM*2)(rnn_layer)
        pooling_layer = keras.layers.GlobalAveragePooling1D()(attention_layer)
        dropout = keras.layers.Dropout(0.5)(pooling_layer)
        y = keras.layers.Dense(1, activation='sigmoid')(dropout)
        model = keras.Model(inputs=inputs, outputs=[y])
        if self.plot:
            keras.utils.plot_model(model, to_file='model/Model10.png', show_shapes=True)
        print(model.summary())
        return model

    def Model10_1(self):# Bilstm step + attention + GobalAvgPooling + dense
        inputs = keras.Input(shape=(self.lens, ), )
        embed = keras.layers.Embedding(input_dim=len(char2idx), output_dim=EMBEDDING_DIM)(inputs)
        rnn_layer = keras.layers.Bidirectional(keras.layers.CuDNNLSTM(EMBEDDING_DIM, return_sequences=True))(embed)
        attention_layer = SeqSelfAttention(attention_activation='sigmoid')(rnn_layer)
        pooling_layer = keras.layers.GlobalAveragePooling1D()(attention_layer)
        dropout = keras.layers.Dropout(0.5)(pooling_layer)
        y = keras.layers.Dense(1, activation='sigmoid')(dropout)
        model = keras.Model(inputs=inputs, outputs=[y])
        if self.plot:
            keras.utils.plot_model(model, to_file='model/Model10.png', show_shapes=True)
        print(model.summary())
        return model

    ####
    def Model12(self):# DPCNN
        filter_nr = 64  # 滤波器通道个数
        filter_size = 3  # 卷积核
        max_pool_size = 3  # 池化层的pooling_size
        max_pool_strides = 2  # 池化层的步长
        dense_nr = 256  # 全连接层
        spatial_dropout = 0.2
        dense_dropout = 0.5
        train_embed = False
        conv_kern_reg = keras.regularizers.l2(0.00001)
        conv_bias_reg = keras.regularizers.l2(0.00001)

        inputs = keras.Input(shape=(self.lens, ), )
        embed = keras.layers.Embedding(input_dim=len(char2idx), output_dim=EMBEDDING_DIM)(inputs)
        embed = keras.layers.SpatialDropout1D(spatial_dropout)(embed)

        block1 = keras.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(embed)
        block1 = keras.layers.BatchNormalization()(block1)
        block1 = keras.layers.PReLU()(block1)
        block1 = keras.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block1)
        block1 = keras.layers.BatchNormalization()(block1)
        block1 = keras.layers.PReLU()(block1)

        # we pass embed through conv1d with filter size 1 because it needs to have the same shape as block output
        # if you choose filter_nr = embed_size (300 in this case) you don't have to do this part and can add emb_comment directly to block1_output
        resize_emb = keras.layers.Conv1D(filter_nr, kernel_size=1, padding='same', activation='linear', kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(embed)
        resize_emb = keras.layers.PReLU()(resize_emb)

        block1_output = keras.layers.add([block1, resize_emb])
        block1_output = keras.layers.MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block1_output)

        block2 = keras.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block1_output)
        block2 = keras.layers.BatchNormalization()(block2)
        block2 = keras.layers.PReLU()(block2)
        block2 = keras.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block2)
        block2 = keras.layers.BatchNormalization()(block2)
        block2 = keras.layers.PReLU()(block2)

        block2_output = keras.layers.add([block2, block1_output])
        block2_output = keras.layers.MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block2_output)

        block3 = keras.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block2_output)
        block3 = keras.layers.BatchNormalization()(block3)
        block3 = keras.layers.PReLU()(block3)
        block3 = keras.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block3)
        block3 = keras.layers.BatchNormalization()(block3)
        block3 = keras.layers.PReLU()(block3)

        block3_output = keras.layers.add([block3, block2_output])
        block3_output = keras.layers.MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block3_output)

        block4 = keras.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block3_output)
        block4 = keras.layers.BatchNormalization()(block4)
        block4 = keras.layers.PReLU()(block4)
        block4 = keras.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block4)
        block4 = keras.layers.BatchNormalization()(block4)
        block4 = keras.layers.PReLU()(block4)

        block4_output = keras.layers.add([block4, block3_output])
        block4_output = keras.layers.MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block4_output)

        block5 = keras.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block4_output)
        block5 = keras.layers.BatchNormalization()(block5)
        block5 = keras.layers.PReLU()(block5)
        block5 = keras.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block5)
        block5 = keras.layers.BatchNormalization()(block5)
        block5 = keras.layers.PReLU()(block5)

        block5_output = keras.layers.add([block5, block4_output])
        block5_output = keras.layers.MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block5_output)

        block6 = keras.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block5_output)
        block6 = keras.layers.BatchNormalization()(block6)
        block6 = keras.layers.PReLU()(block6)
        block6 = keras.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block6)
        block6 = keras.layers.BatchNormalization()(block6)
        block6 = keras.layers.PReLU()(block6)

        block6_output = keras.layers.add([block6, block5_output])
        block6_output = keras.layers.MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block6_output)

        block7 = keras.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block6_output)
        block7 = keras.layers.BatchNormalization()(block7)
        block7 = keras.layers.PReLU()(block7)
        block7 = keras.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block7)
        block7 = keras.layers.BatchNormalization()(block7)
        block7 = keras.layers.PReLU()(block7)

        block7_output = keras.layers.add([block7, block6_output])
        block7_output = keras.layers.GlobalMaxPooling1D()(block7_output)

        outputs = keras.layers.Dense(dense_nr, activation='linear')(block7_output)
        outputs = keras.layers.BatchNormalization()(outputs)
        outputs = keras.layers.PReLU()(outputs)
        outputs = keras.layers.Dropout(dense_dropout)(outputs)
        outputs = keras.layers.Dense(1, activation='sigmoid')(outputs)

        model = keras.Model(inputs=inputs, outputs=outputs)
        if self.plot:
            keras.utils.plot_model(model, to_file='model/Model10.png', show_shapes=True)
        print(model.summary())
        return model


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

class jason_tools(object):
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def train_model(self, model, model_path, original_model):
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        checkpoint = keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', mode='min', verbose=1, save_best_only=True, save_weights_only=1, period=1)
        earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=0, mode='min', restore_best_weights=True)
        model_history = model.fit(self.x_train, self.y_train, shuffle=True, epochs=EPOCHS, validation_split=0.1, callbacks=[checkpoint, earlystop])
        test_loss, test_acc = model.evaluate(self.x_test, self.y_test, verbose=0)
        test_precision, test_recall, test_f1_1 = self.cal_pr(model)
        logging.info('not call back!')
        logging.info('test_loss: %.3f - test_acc: %.3f' % (test_loss, test_acc))
        logging.info('**************************************************')
        logging.info('test_precision: %.3f - test_recall: %.3f - test_f1_score: %.3f' % (test_precision, test_recall, test_f1_1))
        logging.info('**************************************************')

        model = original_model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.load_weights(model_path)
        test_loss, test_acc = model.evaluate(self.x_test, self.y_test, verbose=0)
        test_precision, test_recall, test_f1_1 = self.cal_pr(model)
        #model_history.history['test_precision'] = test_precision
        #model_history.history['test_recall'] = test_recall
        #model_history.history['test_f1'] = test_f1_1

        logging.info('call_back')
        logging.info('test_loss: %.3f - test_acc: %.3f' % (test_loss, test_acc))
        logging.info('**************************************************')
        logging.info('test_precision: %.3f - test_recall: %.3f - test_f1_score: %.3f' % (test_precision, test_recall, test_f1_1))
        logging.info('**************************************************')
        logging.info('best_model_path:  %s' % model_path)
        logging.info('test_accuracy:    %.3f' % test_acc)
        logging.info('test_precision:   %.3f' % test_precision)
        logging.info('test_recall:      %.3f' % test_recall)
        logging.info('test_f1_score:    %.3f' % test_f1_1)
        return model_history
        #return test_f1_1

    def cal_pr(self, model):
        pred = model.predict(self.x_test)
        pred = [i[0] for i in pred]
        pred = [1 if i>=0.5 else 0 for i in pred]
        precision = precision_score(self.y_test, pred)
        recall = recall_score(self.y_test, pred)
        f1 = f1_score(self.y_test, pred)
        logging.info('classification_report: \n')
        logging.info(classification_report(self.y_test, pred))
        logging.info('confusion_matrix: \n')
        logging.info(confusion_matrix(self.y_test, pred))
        return precision, recall, f1

    def plot_history(self, histories, path='model/acc_char.png', key='acc'):
        plt.figure(figsize=(16, 10))

        for name, history in histories:
            val = plt.plot(history.epoch, history.history['val_' + key], '--', label=name.title() + ' Val')
            plt.plot(history.epoch, history.history[key], color=val[0].get_color(), label=name.title() + ' Train')
        plt.xlabel('Epochs')
        plt.ylabel(key.replace('_', ' ').title())
        plt.legend()

        plt.xlim([0, max(history.epoch)])
        plt.savefig(path)

def plot_pr(plist, rlist, f1list, tlist, path):
    plt.figure(figsize=(16, 10))
    plt.plot(range(0, 100), plist, label='Precision')
    plt.plot(range(0, 100), rlist, label='Recall')
    plt.plot(range(0, 100), f1list, label='F1 score')
    plt.xlabel('threshold')
    plt.ylabel('PCT'.title())
    plt.legend()

    plt.xlim([0, len(tlist)])
    plt.savefig(path)

if __name__ == '__main__':
    #############
    # check data
    import pandas as pd

    data_path = './data/tungee/v0_7/Train_gxb_v0.3.tsv'
    train_path = './data/tungee/v0_7/all_gxb_train.tsv'
    test_path  = './data/tungee/v0_7/all_gxb_test.tsv'
    df = pd.read_csv(data_path, sep = '\t', header=None)
    print('中性: ', len(df.loc[df[1] == '中性']))
    print('正面: ', len(df.loc[df[1] == '正面']))
    print('负面: ', len(df.loc[df[1] == '负面']))

    from sklearn.model_selection import train_test_split
    df_train, df_test = train_test_split(df, test_size=0.05, random_state=42)

    print('df_train')
    print('中性: ', len(df_train.loc[df_train[1] == '中性']))
    print('正面: ', len(df_train.loc[df_train[1] == '正面']))
    print('负面: ', len(df_train.loc[df_train[1] == '负面']))
    print('df_test')
    print('中性: ', len(df_test.loc[df_test[1] == '中性']))
    print('正面: ', len(df_test.loc[df_test[1] == '正面']))
    print('负面: ', len(df_test.loc[df_test[1] == '负面']))

    df_train.to_csv(train_path, sep='\t', header=None, index=False)
    df_test.to_csv(test_path, sep='\t', header=None, index=False)
    #############
    # load data
    x_train, y_train = load_data(train_path)
    x_test, y_test = load_data(test_path)

    x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN, padding='post', truncating='post')
    x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN, padding='post', truncating='post')

    x_train = data_transform(x_train, used_fields)
    x_test = data_transform(x_test, used_fields)

    x_train1, y_train_tob, y_train_toc = load_data_new(path=train_path, used_fields=used_fields)
    x_test1, y_test_tob, y_test_toc = load_data(path=test_path, used_fields=used_fields)

    x_train1 = data_transform(x_train1, used_fields)
    x_test1 = data_transform(x_test1, used_fields)

    lens = x_train.shape[1]
    ###########################################################
    '''
    # keras model
    #ToB 4&7 best for ToB
    nlp = nlpModel(lens, plot=False)
    model1 = nlp.Model1()
    model2 = nlp.Model2()
    model3 = nlp.Model3()
    model4 = nlp.Model4()
    model4_1 = nlp.Model4_1()
    model5 = nlp.Model5()
    model5_1 = nlp.Model5_1()
    model5_2 = nlp.Model5_2()
    model5_3 = nlp.Model5_3()
    model5_4 = nlp.Model5_4()
    model5_5 = nlp.Model5_5()
    model6 = nlp.Model6()
    model7 = nlp.Model7()
    model7_1 = nlp.Model7_1()
    model7_2 = nlp.Model7_2()
    model7_3 = nlp.Model7_3()
    model7_4 = nlp.Model7_4()
    model7_5 = nlp.Model7_5()
    model8 = nlp.Model8()
    model9 = nlp.Model9()
    model10 = nlp.Model10()
    model10_1 = nlp.Model10_1()
    model11 = nlp.Model11()
    model12 = nlp.Model12()

    utils = jason_tools(x_train, y_train_tob, x_test, y_test_tob)
    histories = []
    histories.append(('model1', utils.train_model(model1, os.path.join(base_path, 'model/model1_char_tob_weight.h5'), nlp.Model1())))
    histories.append(('model2', utils.train_model(model2, os.path.join(base_path, 'model/model2_char_tob_weight.h5'), nlp.Model2())))
    histories.append(('model3', utils.train_model(model3, os.path.join(base_path, 'model/model3_char_tob_weight.h5'), nlp.Model3())))
    histories.append(('model4', utils.train_model(model4, os.path.join(base_path, 'model/model4_char_tob_weight.h5'), nlp.Model4())))
    histories.append(('model4_1', utils.train_model(model4_1, os.path.join(base_path, 'model/model4_1_char_tob_weight.h5'), nlp.Model4_1())))
    histories.append(('model5', utils.train_model(model5, os.path.join(base_path, 'model/model5_char_tob_weight.h5'), nlp.Model5())))
    histories.append(('model5_1', utils.train_model(model5_1, os.path.join(base_path, 'model/model5_1_char_tob_weight.h5'), nlp.Model5_1())))
    histories.append(('model5_2', utils.train_model(model5_2, os.path.join(base_path, 'model/model5_2_char_tob_weight.h5'), nlp.Model5_2())))
    histories.append(('model5_3', utils.train_model(model5_3, os.path.join(base_path, 'model/model5_3_char_tob_weight.h5'), nlp.Model5_3())))
    histories.append(('model5_4', utils.train_model(model5_4, os.path.join(base_path, 'model/model5_4_char_tob_weight.h5'), nlp.Model5_4())))
    histories.append(('model5_5', utils.train_model(model5_5, os.path.join(base_path, 'model/model5_5_char_tob_weight.h5'), nlp.Model5_5())))
    histories.append(('model6', utils.train_model(model6, os.path.join(base_path, 'model/model6_char_tob_weight.h5'), nlp.Model6())))
    histories.append(('model7', utils.train_model(model7, os.path.join(base_path, 'model/model7_char_tob_weight.h5'), nlp.Model7())))
    histories.append(('model7_1', utils.train_model(model7_1, os.path.join(base_path, 'model/model7_1_char_tob_weight.h5'), nlp.Model7_1())))
    histories.append(('model7_2', utils.train_model(model7_2, os.path.join(base_path, 'model/model7_2_char_tob_weight.h5'), nlp.Model7_2())))
    histories.append(('model7_3', utils.train_model(model7_3, os.path.join(base_path, 'model/model7_3_char_tob_weight.h5'), nlp.Model7_3())))
    histories.append(('model7_4', utils.train_model(model7_4, os.path.join(base_path, 'model/model7_4_char_tob_weight.h5'), nlp.Model7_4())))
    histories.append(('model7_5', utils.train_model(model7_5, os.path.join(base_path, 'model/model7_5_char_tob_weight.h5'), nlp.Model7_5())))
    histories.append(('model8', utils.train_model(model8, os.path.join(base_path, 'model/model8_char_tob_weight.h5'), nlp.Model8())))
    histories.append(('model9', utils.train_model(model9, os.path.join(base_path, 'model/model9_char_tob_weight.h5'), nlp.Model9())))
    histories.append(('model10', utils.train_model(model10, os.path.join(base_path, 'model/model10_char_tob_weight.h5'), nlp.Model10())))
    histories.append(('model10_1', utils.train_model(model10_1, os.path.join(base_path, 'model/model10_1_char_tob_weight.h5'), nlp.Model10_1())))
    histories.append(('model11', utils.train_model(model11, os.path.join(base_path, 'model/model11_char_tob_weight.h5'), nlp.Model11())))
    histories.append(('model12', utils.train_model(model12, os.path.join(base_path, 'model/model12_char_tob_weight.h5'), nlp.Model12())))
    pickle.dump(histories, open('model/model_histories_tob.pickle', 'wb'))
    utils.plot_history(histories, path='model/acc_char_tob.png')

    # ToC 7_1 best for ToC
    nlp = nlpModel(lens, plot=False)
    model1 = nlp.Model1()
    model2 = nlp.Model2()
    model3 = nlp.Model3()
    model4 = nlp.Model4()
    model4_1 = nlp.Model4_1()
    model5 = nlp.Model5()
    model5_1 = nlp.Model5_1()
    model5_2 = nlp.Model5_2()
    model5_3 = nlp.Model5_3()
    model5_4 = nlp.Model5_4()
    model5_5 = nlp.Model5_5()
    model6 = nlp.Model6()
    model7 = nlp.Model7()
    model7_1 = nlp.Model7_1()
    model7_2 = nlp.Model7_2()
    model7_3 = nlp.Model7_3()
    model7_4 = nlp.Model7_4()
    model7_5 = nlp.Model7_5()
    model8 = nlp.Model8()
    model9 = nlp.Model9()
    model10 = nlp.Model10()
    model10_1 = nlp.Model10_1()
    model11 = nlp.Model11()
    model12 = nlp.Model12()

    utils = jason_tools(x_train, y_train_toc, x_test, y_test_toc)
    histories = []
    histories.append(('model1', utils.train_model(model1, os.path.join(base_path, 'model/model1_char_toc_weight.h5'), nlp.Model1())))
    histories.append(('model2', utils.train_model(model2, os.path.join(base_path, 'model/model2_char_toc_weight.h5'), nlp.Model2())))
    histories.append(('model3', utils.train_model(model3, os.path.join(base_path, 'model/model3_char_toc_weight.h5'), nlp.Model3())))
    histories.append(('model4', utils.train_model(model4, os.path.join(base_path, 'model/model4_char_toc_weight.h5'), nlp.Model4())))
    histories.append(('model4_1', utils.train_model(model4_1, os.path.join(base_path, 'model/model4_1_char_toc_weight.h5'), nlp.Model4_1())))
    histories.append(('model5', utils.train_model(model5, os.path.join(base_path, 'model/model5_char_toc_weight.h5'), nlp.Model5())))
    histories.append(('model5_1', utils.train_model(model5_1, os.path.join(base_path, 'model/model5_1_char_toc_weight.h5'), nlp.Model5_1())))
    histories.append(('model5_2', utils.train_model(model5_2, os.path.join(base_path, 'model/model5_2_char_toc_weight.h5'), nlp.Model5_2())))
    histories.append(('model5_3', utils.train_model(model5_3, os.path.join(base_path, 'model/model5_3_char_toc_weight.h5'), nlp.Model5_3())))
    histories.append(('model5_4', utils.train_model(model5_4, os.path.join(base_path, 'model/model5_4_char_toc_weight.h5'), nlp.Model5_4())))
    histories.append(('model5_5', utils.train_model(model5_5, os.path.join(base_path, 'model/model5_5_char_toc_weight.h5'), nlp.Model5_5())))
    histories.append(('model6', utils.train_model(model6, os.path.join(base_path, 'model/model6_char_toc_weight.h5'), nlp.Model6())))
    histories.append(('model7', utils.train_model(model7, os.path.join(base_path, 'model/model7_char_toc_weight.h5'), nlp.Model7())))
    histories.append(('model7_1', utils.train_model(model7_1, os.path.join(base_path, 'model/model7_1_char_toc_weight.h5'), nlp.Model7_1())))
    histories.append(('model7_2', utils.train_model(model7_2, os.path.join(base_path, 'model/model7_2_char_toc_weight.h5'), nlp.Model7_2())))
    histories.append(('model7_3', utils.train_model(model7_3, os.path.join(base_path, 'model/model7_3_char_toc_weight.h5'), nlp.Model7_3())))
    histories.append(('model7_4', utils.train_model(model7_4, os.path.join(base_path, 'model/model7_4_char_toc_weight.h5'), nlp.Model7_4())))
    histories.append(('model7_5', utils.train_model(model7_5, os.path.join(base_path, 'model/model7_5_char_toc_weight.h5'), nlp.Model7_5())))
    histories.append(('model8', utils.train_model(model8, os.path.join(base_path, 'model/model8_char_toc_weight.h5'), nlp.Model8())))
    histories.append(('model9', utils.train_model(model9, os.path.join(base_path, 'model/model9_char_toc_weight.h5'), nlp.Model9())))
    histories.append(('model10', utils.train_model(model10, os.path.join(base_path, 'model/model10_char_toc_weight.h5'), nlp.Model10())))
    histories.append(('model10_1', utils.train_model(model10_1, os.path.join(base_path, 'model/model10_1_char_toc_weight.h5'), nlp.Model10_1())))
    histories.append(('model11', utils.train_model(model11, os.path.join(base_path, 'model/model11_char_toc_weight.h5'), nlp.Model11())))
    histories.append(('model12', utils.train_model(model12, os.path.join(base_path, 'model/model12_char_toc_weight.h5'), nlp.Model12())))
    pickle.dump(histories, open('model/model_histories_toc.pickle', 'wb'))
    utils.plot_history(histories, path='model/acc_char_toc.png')
    '''




