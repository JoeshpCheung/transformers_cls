# *-* coding:utf-8 *-*
"""
    Description:
    ~~~~~~~~~~~~~~~~~~~~~~~
    @project: alg-emotion-train
    @author: JasonCheung
    @file: general_sentiment_train_1model.py
    @editor: PyCharm
    @time: 2020-02-27 10:06:18

"""

# *-* coding:utf-8 *-*
"""
    Description:
    ~~~~~~~~~~~~~~~~~~~~~~~
    @project: alg-emotion-train
    @author: JasonCheung
    @file: general_sentiment_train.py
    @editor: PyCharm
    @time: 2020-01-09 16:24:53

"""

import logging
import pickle
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

# 读取词向量相关文件
vector_path = 'word2vec/sgns.zhihu.word'

wrd2idx = {}
idx2wrd = {}

wrd2idx['BNK'] = 0
wrd2idx['<UNK>'] = 1
idx2wrd[0] = 'BNK'
idx2wrd[1] = '<UNK>'

# 用词向量的话
embeddings_matrix = []
embeddings_matrix = np.array(embeddings_matrix)
EMBEDDING_DIM_WORD = 100

class nlpModel(object):
    def __init__(self, lens, embed=None, plot=True):
        self.lens = lens
        self.plot = plot
        self.embeddings_matrix = embed

    def model1(self):  # fast_text
        inputs = keras.Input(shape=(self.lens,), )
        mask = keras.layers.Masking(mask_value=0)(inputs)
        if not self.embeddings_matrix is None:
            embed_dim = EMBEDDING_DIM_WORD
            embed = keras.layers.Embedding(input_dim=len(wrd2idx), output_dim=embed_dim,
                                           weights=[self.embeddings_matrix], trainable=False)(mask)
        else:
            embed_dim = EMBEDDING_DIM_CHAR
            embed = keras.layers.Embedding(input_dim=len(char2idx), output_dim=embed_dim)(mask)
        pooling_layer = keras.layers.GlobalAveragePooling1D()(embed)
        dropout_layer = keras.layers.Dropout(0.3)(pooling_layer)
        y = keras.layers.Dense(2, activation='softmax')(dropout_layer)
        model = keras.Model(inputs=inputs, outputs=[y])
        if self.plot:
            keras.utils.plot_model(model, to_file='model/Model1.png', show_shapes=True)
        return model

    def model2(self):  # CNN
        inputs = keras.Input(shape=(self.lens,), )
        mask = keras.layers.Masking(mask_value=0)(inputs)
        if not self.embeddings_matrix is None:
            embed_dim = EMBEDDING_DIM_WORD
            embed = keras.layers.Embedding(input_dim=len(wrd2idx), output_dim=embed_dim,
                                           weights=[self.embeddings_matrix], trainable=False)(mask)
        else:
            embed_dim = EMBEDDING_DIM_CHAR
            embed = keras.layers.Embedding(input_dim=len(char2idx), output_dim=embed_dim)(mask)
        cnn_layer = keras.layers.Conv1D(256, 5, padding='same', strides=1, activation='relu')(embed)
        pooling_layer = keras.layers.MaxPool1D(pool_size=46)(cnn_layer)
        flatten_layer = keras.layers.Flatten()(pooling_layer)
        dropout = keras.layers.Dropout(0.3)(flatten_layer)
        y = keras.layers.Dense(1, activation='sigmoid')(dropout)
        model = keras.Model(inputs=inputs, outputs=[y])
        if self.plot:
            keras.utils.plot_model(model, to_file='model/Model2.png', show_shapes=True)
        print(model.summary())
        return model

    def model3(self):  # CNN (LeNet-5)
        inputs = keras.Input(shape=(self.lens,), )
        mask = keras.layers.Masking(mask_value=0)(inputs)
        if not self.embeddings_matrix is None:
            embed_dim = EMBEDDING_DIM_WORD
            embed = keras.layers.Embedding(input_dim=len(wrd2idx), output_dim=embed_dim,
                                           weights=[self.embeddings_matrix], trainable=False)(mask)
        else:
            embed_dim = EMBEDDING_DIM_CHAR
            embed = keras.layers.Embedding(input_dim=len(char2idx), output_dim=embed_dim)(mask)
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

        y = keras.layers.Dense(1, activation='sigmoid')(dropout)
        model = keras.Model(inputs=inputs, outputs=[y])
        if self.plot:
            keras.utils.plot_model(model, to_file='model/Model2.png', show_shapes=True)
        print(model.summary())
        return model

    def model4(self):  # combine input - textcnn - flatten - dense1 # best negetive_f1: 0.78, acc: 0.892
        inputs = keras.Input(shape=(self.lens,), )
        mask = keras.layers.Masking(mask_value=0)(inputs)
        if not self.embeddings_matrix is None:
            embed_dim = EMBEDDING_DIM_WORD
            embed = keras.layers.Embedding(input_dim=len(wrd2idx), output_dim=embed_dim,
                                           weights=[self.embeddings_matrix], trainable=False)(mask)
        else:
            embed_dim = EMBEDDING_DIM_CHAR
            embed = keras.layers.Embedding(input_dim=len(char2idx), output_dim=embed_dim)(mask)

        cnn1 = keras.layers.Conv1D(256, 3, padding='same', strides=1, activation='relu')(embed)
        cnn1 = keras.layers.MaxPooling1D(pool_size=38)(cnn1)
        cnn2 = keras.layers.Conv1D(256, 4, padding='same', strides=1, activation='relu')(embed)
        cnn2 = keras.layers.MaxPooling1D(pool_size=37)(cnn2)
        cnn3 = keras.layers.Conv1D(256, 5, padding='same', strides=1, activation='relu')(embed)
        cnn3 = keras.layers.MaxPooling1D(pool_size=36)(cnn3)

        cnn = keras.layers.concatenate([cnn1, cnn2, cnn3], axis=-1)
        flat = keras.layers.Flatten()(cnn)
        drop = keras.layers.Dropout(0.5)(flat)
        y = keras.layers.Dense(3, activation='softmax')(drop)
        model = keras.Model(inputs=inputs, outputs=[y])
        if self.plot:
            keras.utils.plot_model(model, to_file='model/Model3.png', show_shapes=True)
        print(model.summary())
        return model


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
        earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, verbose=0, mode='min',
                                                  restore_best_weights=True)
        model_history = model.fit(self.x_train, self.y_train, shuffle=True, epochs=EPOCHS, validation_split=0.1,
                                  callbacks=[checkpoint, earlystop], class_weight=class_weight)

        model = original_model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
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


def predict(model, string):
    string = extract(string)
    string = string.replace(' ', '').replace('\t', '').strip()
    string = [char2idx[i] if i in char2idx else char2idx['<UNK>'] for i in string]
    string = sequence.pad_sequences([string], maxlen=MAX_LEN, padding='post', truncating='post')
    res = model.predict(string)
    return res


# 文本预处理规则
def extract(s):
    result = re.sub('回复(.*?):', '', s)
    return result


def corpus2label(datas):
    punc = r'~`!#$%^&*()_+-=|\\\';":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}'
    datas = [extract(i) for i in datas]
    datas = [i.replace(' ', '').replace('\t', '').strip() for i in datas]
    datas = [re.sub(r"[%s]+" % punc, "", i) for i in datas]
    datas = [[char2idx[i] if i in char2idx else char2idx['<UNK>'] for i in j] for j in datas]
    datas = sequence.pad_sequences(datas, maxlen=MAX_LEN, padding='post', truncating='post')
    print('datas.shape', datas.shape)

    return datas


if __name__ == '__main__':
    print('start')
    path_sina = './data/re-annotation/weibo_senti_100k.csv'
    df_sina = pd.read_csv(path_sina)
    df_sina = shuffle(df_sina)

    # # 将data转换成test/train dataset
    # path = './data/re-annotation/data.xlsx'
    # df = pd.read_excel(path)
    # df.dropna(how='all', subset=['positive', 'neutral', 'negative'], inplace=True)
    # df.fillna(value=0, inplace=True)
    # df['cmnt'] = [i.replace('\t', ' ').replace('\n', ' ').strip() for i in df['cmnt'].tolist()]
    # df['positive'] = [int(i) for i in df['positive'].tolist()]
    # df['neutral'] = [int(i) for i in df['neutral'].tolist()]
    # df['negative'] = [int(i) for i in df['negative'].tolist()]
    #
    # train, test = train_test_split(df, test_size=0.1)
    # train.to_excel('./data/re-annotation/train.xlsx', index=False)
    # test.to_excel('./data/re-annotation/test.xlsx', index=False)
    #
    # path_positive = '/home/tungee/桌面/jasoncheung/工信部/alg-emotion-train/data/re-annotation/positive.txt'
    # path_neutral = '/home/tungee/桌面/jasoncheung/工信部/alg-emotion-train/data/re-annotation/neutral.txt'
    # path_negative = '/home/tungee/桌面/jasoncheung/工信部/alg-emotion-train/data/re-annotation/negative.txt'
    # df_negative = train[['negative', 'cmnt']]
    # df_neutral = train[['neutral', 'cmnt']]
    # df_positive = train[['positive', 'cmnt']]
    #
    # df_negative.to_csv(path_negative, sep='\t', index=False, header=False)
    # df_neutral.to_csv(path_neutral, sep='\t', index=False, header=False)
    # df_positive.to_csv(path_positive, sep='\t', index=False, header=False)

    train = pd.read_excel('./data/re-annotation/train_independent.xlsx')
    test = pd.read_excel('./data/re-annotation/test_independent.xlsx')

    tmp = '回复@云姉:可以啊，手机号给我，我天天打，问你有没有吃饭啊，有没有吃药啊，好不好啊，我的电话费你出，如何？'

    # 文本预处理规则
    def extract(s):
        result = re.sub('回复@(.*?):', '', s)
        return result


    # # use EDA
    # path_positive = '/home/tungee/桌面/jasoncheung/工信部/alg-emotion-train/data/re-annotation/positive_rd_sr_8.txt'
    # path_neutral = '/home/tungee/桌面/jasoncheung/工信部/alg-emotion-train/data/re-annotation/neutral_rd_sr_8.txt'
    # path_negative = '/home/tungee/桌面/jasoncheung/工信部/alg-emotion-train/data/re-annotation/negative_rd_sr_8.txt'
    # df_negative = pd.read_csv(path_negative, sep='\t', header=None)
    # df_neutral = pd.read_csv(path_neutral, sep='\t', header=None)
    # df_positive = pd.read_csv(path_positive, sep='\t', header=None)
    # train = pd.DataFrame(data={'cmnt': df_negative[1],
    #                         'positive': df_positive[0],
    #                         'neutral': df_neutral[0],
    #                         'negative': df_negative[0]})
    # train = train.dropna()
    # train = pd.DataFrame(data={'cmnt': df_negative[1],
    #                         'positive': df_negative[0],
    #                         'neutral': df_negative[0],
    #                         'negative': df_negative[0]})

    # positive 正负数量差距过大!导致模型无法学习有效特征
    # get train data & label
    x_train = train['cmnt'].tolist()
    x_train = corpus2label(x_train)

    y_train_pos = train['positive'].tolist()
    y_train_neu = train['neutral'].tolist()
    y_train_neg = train['negative'].tolist()
    y_train_pos = [int(i) for i in y_train_pos]
    y_train_neu = [int(i) for i in y_train_neu]
    y_train_neg = [int(i) for i in y_train_neg]

    y_train = [[y_train_pos[i], y_train_neu[i], y_train_neg[i]] for i in range(len(y_train_pos))]

    # get test data & label
    x_test = test['cmnt'].tolist()
    x_test = [extract(i) for i in x_test]
    x_test = [i.replace(' ', '').replace('\t', '').strip() for i in x_test]

    y_test_positive = test['positive'].tolist()
    y_test_neutral = test['neutral'].tolist()
    y_test_negative = test['negative'].tolist()
    y_test_pos = [int(i) for i in y_test_positive]
    y_test_neu = [int(i) for i in y_test_neutral]
    y_test_neg = [int(i) for i in y_test_negative]
    y_test = [[y_test_pos[i], y_test_neu[i], y_test_neg[i]] for i in range(len(y_test_pos))]

    # y_test_positive = keras.utils.to_categorical(y_test_positive)
    # y_test_neutral = keras.utils.to_categorical(y_test_neutral)
    # y_test_negative = keras.utils.to_categorical(y_test_negative)

    # transfer data
    x_train_char = [[char2idx[i] if i in char2idx else char2idx['<UNK>'] for i in j] for j in x_train]
    x_test_char = [[char2idx[i] if i in char2idx else char2idx['<UNK>'] for i in j] for j in x_test]
    # x_test_positive_char = [[char2idx[i] if i in char2idx else char2idx['<UNK>'] for i in j] for j in x_test_positive]

    print('padding sequences')
    x_train_char = sequence.pad_sequences(x_train_char, maxlen=MAX_LEN, padding='post', truncating='post')
    x_test_char = sequence.pad_sequences(x_test_char, maxlen=MAX_LEN, padding='post', truncating='post')

    print('x_train_char shape:', x_train_char.shape)
    print('x_test_char shape:', x_test_char.shape)

    # train model 验证模型发现cnn 与 textcnn性价比最高
    nlp = nlpModel(MAX_LEN, plot=False)
    model_path = 'model/model-textcnn_char_all.h5'

    y_train = np.array([np.array(i) for i in y_train])
    y_test = np.array([np.array(i) for i in y_test])

    # train model
    class_weight = {0: 5, 1: 1, 2: 1}
    model4 = nlp.model4()  # CNN (LeNet-5)
    utils_negative = JasonTools(x_train_char, y_train, x_test_char, y_test)
    record_negative = []
    record_negative.append(('model4-textcnn',
                            utils_negative.train_model(model4, model_path,
                                                       nlp.model4())))
    utils_negative.plot_history(record_negative, path='model/acc_char_emotion.png')
