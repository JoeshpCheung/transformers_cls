# *-* coding:utf-8 *-*
"""
    Description:
    ~~~~~~~~~~~~~~~~~~~~~~~
    @project: alg-emotion-train
    @author: JasonCheung
    @file: my_common_train.py
    @editor: PyCharm
    @time: 2019-12-27 17:03:00

"""

import keras
import pickle
import logging
import jieba
import numpy as np
import pandas as pd
import gensim
import tensorflow as tf
from keras import backend as K
import matplotlib.pyplot as plt
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

logging.basicConfig(filename='log/train_model_char.log', filemode='a+', level=logging.INFO, format='%(message)s')

MAX_LEN = 100
EPOCHS = 300
EMBEDDING_DIM_CHAR = 100

# 读取字符向量相关文件
char_dict_path = 'data/tungee/v0_7/char_vocabs.pkl'
char_vocabs = pickle.load(open(char_dict_path, 'rb'))
special_words = ['BNK', '<UNK>', '<PAD>']
char_vocabs = special_words + char_vocabs

idx2char = {idx: char for idx, char in enumerate(char_vocabs)}
char2idx = {char: idx for idx, char in idx2char.items()}

# 读取词向量相关文件
vector_path = 'word2vec/sgns.zhihu.word'

wrd2idx = {}
idx2wrd = {}
wrd2idx['BNK'] = 0
wrd2idx['<UNK>'] = 1
idx2wrd[0] = 'BNK'
idx2wrd[1] = '<UNK>'

embeddings_matrix = []
# 用词向量的话
word_dim = dock100

embeddings_matrix = np.array(embeddings_matrix)
EMBEDDING_DIM_WORD = word_dim

class nlpModel(object):
    def __init__(self, lens, embed=None, plot=True):
        self.lens = lens
        self.plot = plot
        self.embeddings_matrix = embed

    def Model1(self):# fast_text
        inputs = keras.Input(shape=(self.lens, ), )
        mask = keras.layers.Masking(mask_value=0)(inputs)
        if type(self.embeddings_matrix) != type(None):
            EMBEDDING_DIM = EMBEDDING_DIM_WORD
            embed = keras.layers.Embedding(input_dim=len(wrd2idx), output_dim=EMBEDDING_DIM, weights=[self.embeddings_matrix], trainable=False)(mask)
        else:
            EMBEDDING_DIM = EMBEDDING_DIM_CHAR
            embed = keras.layers.Embedding(input_dim=len(char2idx), output_dim=EMBEDDING_DIM)(mask)
        pooling_layer = keras.layers.GlobalAveragePooling1D()(embed)
        dropout_layer = keras.layers.Dropout(0.3)(pooling_layer)
        y = keras.layers.Dense(3, activation='softmax')(dropout_layer)
        model = keras.Model(inputs=inputs, outputs=[y])
        if self.plot:
            keras.utils.plot_model(model, to_file='model/Model1.png', show_shapes=True)
        return model

    def Model2(self):# CNN
        inputs = keras.Input(shape=(self.lens, ), )
        mask = keras.layers.Masking(mask_value=0)(inputs)
        if type(self.embeddings_matrix) != type(None):
            EMBEDDING_DIM = EMBEDDING_DIM_WORD
            embed = keras.layers.Embedding(input_dim=len(wrd2idx), output_dim=EMBEDDING_DIM, weights=[self.embeddings_matrix], trainable=False)(mask)
        else:
            EMBEDDING_DIM = EMBEDDING_DIM_CHAR
            embed = keras.layers.Embedding(input_dim=len(char2idx), output_dim=EMBEDDING_DIM)(mask)
        cnn_layer = keras.layers.Conv1D(256, 5, padding='same', strides=1, activation='relu')(embed)
        pooling_layer = keras.layers.MaxPool1D(pool_size=46)(cnn_layer)
        flatten_layer = keras.layers.Flatten()(pooling_layer)
        dropout = keras.layers.Dropout(0.3)(flatten_layer)
        y = keras.layers.Dense(3, activation='softmax')(dropout)
        model = keras.Model(inputs=inputs, outputs=[y])
        if self.plot:
            keras.utils.plot_model(model, to_file='model/Model2.png', show_shapes=True)
        print(model.summary())
        return model

    def Model2_1(self):# CNN (LeNet-5)
        inputs = keras.Input(shape=(self.lens, ), )
        mask = keras.layers.Masking(mask_value=0)(inputs)
        if type(self.embeddings_matrix) != type(None):
            EMBEDDING_DIM = EMBEDDING_DIM_WORD
            embed = keras.layers.Embedding(input_dim=len(wrd2idx), output_dim=EMBEDDING_DIM, weights=[self.embeddings_matrix], trainable=False)(mask)
        else:
            EMBEDDING_DIM = EMBEDDING_DIM_CHAR
            embed = keras.layers.Embedding(input_dim=len(char2idx), output_dim=EMBEDDING_DIM)(mask)
        cnn_layer = keras.layers.Conv1D(256, 5, padding='same', strides=1)(embed)
        pooling_layer = keras.layers.MaxPooling1D(pool_size=3, strides=3, padding='same')(cnn_layer)
        cnn_layer = keras.layers.Conv1D(128, 5, padding='same', strides=1)(pooling_layer)
        pooling_layer = keras.layers.MaxPooling1D(pool_size=3, strides=3, padding='same')(cnn_layer)
        cnn_layer = keras.layers.Conv1D(64, 5, padding='same', strides=1)(pooling_layer)
        pooling_layer = keras.layers.MaxPooling1D(pool_size=3, strides=3, padding='same')(cnn_layer)

        flatten_layer = keras.layers.Flatten()(pooling_layer)
        dropout = keras.layers.Dropout(0.5)(flatten_layer)
        normalize_layer = keras.layers.BatchNormalization()(dropout)
        dense_layer = keras.layers.Dense(256, activation='relu')(normalize_layer)
        dropout = keras.layers.Dropout(0.5)(dense_layer)

        y = keras.layers.Dense(3, activation='softmax')(dropout)
        model = keras.Model(inputs=inputs, outputs=[y])
        if self.plot:
            keras.utils.plot_model(model, to_file='model/Model2.png', show_shapes=True)
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
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        checkpoint = keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', mode='min', verbose=1, save_best_only=True, save_weights_only=1, period=1)
        earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, verbose=0, mode='min', restore_best_weights=True)
        model_history = model.fit(self.x_train, self.y_train, shuffle=True, epochs=EPOCHS, validation_split=0.1, callbacks=[checkpoint, earlystop])

        model = original_model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.load_weights(model_path)
        test_loss, test_acc = model.evaluate(self.x_test, self.y_test, verbose=0)
        report = self.cal_pr(model)

        logging.info('call_back')
        logging.info('test_loss: %.3f - test_acc: %.3f' % (test_loss, test_acc))
        logging.info('**************************************************')
        logging.info(report)
        logging.info('**************************************************')
        logging.info('best_model_path:  %s' % model_path)
        logging.info('test_accuracy:    %.3f' % test_acc)
        return model_history

    def cal_pr(self, model):
        pred = model.predict(self.x_test)
        pred = list(np.argmax(pred, axis=1))
        true = list(np.argmax(self.y_test, axis=1))
        report = classification_report(true, pred)
        logging.info('classification_report: \n')
        logging.info(report)
        logging.info('confusion_matrix: \n')
        logging.info(confusion_matrix(true, pred))
        return report

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

###########################################


if __name__ == '__main__':
    print('start')

    path = '/home/tungee/桌面/jasoncheung/Jason-NLP/emotion-analysis/data/weibo_senti_100k/weibo_senti_100k.csv'
    df = pd.read_csv(path)
    df['label'] = df['label'].replace(1, 2).replace(0, 1)
    train, test = train_test_split(df, test_size=0.05)

    x_train = train['review'].tolist()
    y_train = train['label'].tolist()
    y_train = [int(i) for i in y_train]

    x_test = test['review'].tolist()
    y_test = test['label'].tolist()
    y_test = [int(i) for i in y_test]

    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)
    ###########################################

    ###########################################
    # char model # 验证模型发现cnn 与 textcnn性价比最高
    # transfer data
    x_train_char = [[char2idx[i] if i in char2idx else char2idx['<UNK>'] for i in j] for j in x_train]
    x_test_char = [[char2idx[i] if i in char2idx else char2idx['<UNK>'] for i in j] for j in x_test]

    # model
    print('padding sequences')
    x_train_char = sequence.pad_sequences(x_train_char, maxlen=MAX_LEN, padding='post', truncating='post')
    x_test_char = sequence.pad_sequences(x_test_char, maxlen=MAX_LEN, padding='post', truncating='post')

    print('x_train_char shape:', x_train_char.shape)
    print('x_test_char shape:', x_test_char.shape)

    print('y_train shape:', y_train.shape)
    print('y_test shape:', y_test.shape)

    nlp = nlpModel(MAX_LEN, plot=False)
    # model1 = nlp.Model1()
    model2 = nlp.Model2_1()  # cnn
    # model2.load_weights('/home/tungee/桌面/jasoncheung/工信部/alg-emotion-train/model/model2_char_v0.5.h5')

    #
    # faq_model_pkl = '/home/tungee/桌面/jasoncheung/工信部/alg-emotion/model/model2_char.h5'
    # model = nlpModel(MAX_LEN, plot=False).Model2()
    # model.load_weights(faq_model_pkl)
    # model3 = nlp.Model3()  # textcnn
    # model4 = nlp.Model4()
    # model5 = nlp.Model5()
    # model6 = nlp.Model6()

    utils = jason_tools(x_train_char, y_train, x_test_char, y_test)
    histories = []
    # histories.append(('model1', utils.train_model(model1, 'model/model1_char.h5', nlp.Model1())))
    # histories.append(('model2', utils.train_model(model2, 'model/model2(LeNet-5)_char_v0.5.h5', nlp.Model2())))
    histories.append(('model3', utils.train_model(model2, 'model/model2-lenet5_char_v0.7_common.h5', nlp.Model2_1())))
    # histories.append(('model4', utils.train_model(model4, 'model/model4_char.h5', nlp.Model4())))
    # histories.append(('model5', utils.train_model(model5, 'model/model5_char.h5', nlp.Model5())))
    # histories.append(('model6', utils.train_model(model6, 'model/model6_char.h5', nlp.Model6())))
    utils.plot_history(histories, path='model/acc_char_emotion_common.png')
    print(utils.cal_pr(model2))