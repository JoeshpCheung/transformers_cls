# *-* coding:utf-8 *-*
"""
    Description:
    ~~~~~~~~~~~~~~~~~~~~~~~
    @project: alg-emotion-train
    @author: JasonCheung
    @file: train_model_char_path4.py
    @editor: PyCharm
    @time: 2020-03-02 14:26:27

"""
import logging
import pickle
import time
import codecs
import jieba
import re
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.preprocessing import sequence
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from attention_keras import *
from sklearn.utils import shuffle

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

logging.basicConfig(filename='log/general_sentiment_train.log', filemode='a+', level=logging.INFO, format='%(message)s')

MAX_LEN = 100
EPOCHS = 50
EMBEDDING_DIM_CHAR = 100

# 读取字符向量相关文件
char_dict_path = 'data/tungee/v0_7/char_vocabs.pkl'
char_vocabs = pickle.load(open(char_dict_path, 'rb'))
special_words = ['BNK', '<UNK>', '<PAD>']
char_vocabs = special_words + char_vocabs

idx2char = {idx: char for idx, char in enumerate(char_vocabs)}
char2idx = {char: idx for idx, char in idx2char.items()}


def f1(true, pred):
    def get_recall():
        true_positives = K.sum(K.round(K.clip(true * pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def get_precision():
        true_positives = K.sum(K.round(K.clip(true * pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = get_precision()
    recall = get_recall()
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def binary_focal_loss(gamma=2, alpha=0.25):
    """
    Binary form of focal loss.
    适用于二分类问题的focal loss

    focal_loss(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
        where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        return K.mean(focal_loss)

    return binary_focal_loss_fixed


class JasonTools(object):
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def train_model(self, model, model_path, original_model):
        # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.compile(optimizer='adam', loss=binary_focal_loss(gamma=2, alpha=0.25), metrics=['accuracy'])
        checkpoint = keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', mode='min', verbose=1,
                                                     save_best_only=True, save_weights_only=1, period=1)
        earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min',
                                                  restore_best_weights=True)
        model_history = model.fit(self.x_train, self.y_train, shuffle=True, epochs=EPOCHS, validation_split=0.1,
                                  callbacks=[checkpoint, earlystop])

        model = original_model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.load_weights(model_path)
        test_loss, test_acc = model.evaluate(self.x_test, self.y_test, verbose=0)

        logging.info('best_model_path:  %s' % model_path)
        logging.info('test_loss: %.3f - test_acc: %.3f' % (test_loss, test_acc))
        self.cal_pr(model, self.x_test, self.y_test)

        return model_history

    def finetune_model(self, model, model_path, x_datas, y_datas, class_weight=None):
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        checkpoint = keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss',
                                                     mode='min', verbose=1,
                                                     save_best_only=True, save_weights_only=1, period=1)
        earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, verbose=0, mode='min',
                                                  restore_best_weights=True)
        if class_weight:
            model.fit(x_datas, y_datas, shuffle=True, epochs=EPOCHS,
                      validation_split=0.1,
                      callbacks=[checkpoint, earlystop], class_weight=class_weight)
        else:
            model.fit(x_datas, y_datas, shuffle=True, epochs=EPOCHS,
                      validation_split=0.1,
                      callbacks=[checkpoint, earlystop])

        model.load_weights(model_path)
        test_loss, test_acc = model.evaluate(self.x_test, self.y_test, verbose=0)
        logging.info('best_finetune_model_path:  %s' % model_path)
        logging.info('best_model_path:  %s' % model_path)
        logging.info('test_loss: %.3f - test_acc: %.3f' % (test_loss, test_acc))
        self.cal_pr(model, self.x_test, self.y_test)

    @staticmethod
    def cal_pr(model, x_test, y_test):
        pred = model.predict(x_test)
        pred = [i[0] for i in pred]
        pred = [1 if i >= 0.5 else 0 for i in pred]
        true = y_test
        # pred = list(np.argmax(pred, axis=1))
        # true = list(np.argmax(self.y_test, axis=1))
        report = classification_report(true, pred)
        logging.info('classification_report: \n')
        logging.info(report)
        logging.info('confusion_matrix: \n')
        logging.info(confusion_matrix(true, pred))
        tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
        logging.info('tn: %d, fp: %d, fn: %d, tp: %d' % (tn, fp, fn, tp))
        return report

    @staticmethod
    def plot_history(histories, path='model/acc_char.png', key='acc'):
        plt.figure(figsize=(16, 10))

        for name, history in histories:
            val = plt.plot(history.epoch, history.history['val_' + key], '--', label=name.title() + ' Val')
            plt.plot(history.epoch, history.history[key], color=val[0].get_color(), label=name.title() + ' Train')
        plt.xlabel('Epochs')
        plt.ylabel(key.replace('_', ' ').title())
        plt.legend()

        plt.xlim([0, max(history.epoch)])
        plt.savefig(path)


def init_jieba(user_dict):
    print('start init_jieba() function.')
    tic = time.time()
    for idx, line in enumerate(user_dict):
        w = line.strip()
        if w:
            jieba.add_word(w)
            jieba.suggest_freq(w, tune=True)

    toc = time.time()
    print('init_jieba() add %d custom words.' % (idx+1))
    print('init_jieba() function time use: %.3f' % (toc - tic))
    print('finish init_jieba() function.')


# 文本预处理规则
def extract(s):
    result = re.sub('回复(.*?):', '', s)
    return result


def corpus2label(datas):
    punc = r'~`!#$%^&*()_+-=|\\\';":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}'
    datas = [extract(i) for i in datas]
    # datas = [i.replace(' ', '').replace('\t', '').strip() for i in datas]
    # datas = [re.sub(r"[%s]+" % punc, "", i) for i in datas]
    print('datas.shape', len(datas))
    datas = [list(i) for i in datas]
    # datas = sequence.pad_sequences(datas, maxlen=MAX_LEN, padding='post', truncating='post')
    return datas


if __name__ == '__main__':

    print('start')
    path_sina = './data/re-annotation/weibo_senti_100k.csv'
    df_sina = pd.read_csv(path_sina)
    df_sina = shuffle(df_sina)

    train = pd.read_excel('./data/re-annotation/train_independent.xlsx')
    test = pd.read_excel('./data/re-annotation/test_independent.xlsx')

    tmp = '回复@云姉:可以啊，手机号给我，我天天打，问你有没有吃饭啊，有没有吃药啊，好不好啊，我的电话费你出，如何？'

    x_train = train['cmnt'].tolist()
    x_train = corpus2label(x_train)

    y_train_pos = train['positive'].tolist()
    y_train_neu = train['neutral'].tolist()
    y_train_neg = train['negative'].tolist()
    y_train_pos = [int(i) for i in y_train_pos]
    y_train_neu = [int(i) for i in y_train_neu]
    y_train_neg = [int(i) for i in y_train_neg]

    y_train = [[y_train_pos[i], y_train_neu[i], y_train_neg[i]] for i in range(len(y_train_pos))]
    y_train = np.argmax(y_train, axis=1)

    # get test data & label
    x_test = test['cmnt'].tolist()
    x_test = corpus2label(x_test)

    y_test_positive = test['positive'].tolist()
    y_test_neutral = test['neutral'].tolist()
    y_test_negative = test['negative'].tolist()
    y_test_pos = [int(i) for i in y_test_positive]
    y_test_neu = [int(i) for i in y_test_neutral]
    y_test_neg = [int(i) for i in y_test_negative]
    y_test = [[y_test_pos[i], y_test_neu[i], y_test_neg[i]] for i in range(len(y_test_pos))]
    y_test = np.argmax(y_test, axis=1)

    # bert mdoel
    import kashgari
    import pandas as pd
    from kashgari.embeddings import BERTEmbedding
    from kashgari.tasks.classification import R_CNN_Model
    from kashgari.processors import ClassificationProcessor
    # load bert model
    model_path = './model/bert_base_bilstm_32'
    bert_model = './model/chinese_L-12_H-768_A-12'
    bert_embed = BERTEmbedding(bert_model,
                               task=kashgari.CLASSIFICATION,
                               sequence_length=128)

    model = R_CNN_Model(bert_embed)

    def train(model, datas, labels):
        train_datas, valid_datas, train_labels, valid_labels = train_test_split(datas, labels, test_size=0.05, random_state=22)

        print('train', len(train_datas), len(train_labels))
        print('test', len(valid_datas), len(valid_labels))
        tf_board_callback = keras.callbacks.TensorBoard(log_dir='./logs', update_freq=1000)
        checkpoint = keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', mode='min', verbose=1, save_best_only=True,
                                                     save_weights_only=1, period=1)
        earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='min',
                                                  restore_best_weights=True)
        print('start fit model')
        model.fit(train_datas,
                  train_labels,
                  x_validate=valid_datas,
                  y_validate=valid_labels,
                  epochs=2
                  )

        model.save(model_path)
    y_train = [str(i) for i in y_train]
    y_test = [str(i) for i in y_test]
    train(model, x_train, y_train)
    pred = []
    for idx, i in enumerate(x_test):
        try:
            tmp = model.predict([i])
        except:
            tmp = []
        print(idx, tmp)
        pred = pred.append(tmp)
    pred = model.predict(x_test)
    print(classification_report(y_test, pred))
