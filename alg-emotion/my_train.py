# *-* coding:utf-8 *-*
"""
    Description:
    ~~~~~~~~~~~~~~~~~~~~~~~
    @project: alg-emotion-train
    @author: JasonCheung
    @file: my_train.py
    @editor: PyCharm
    @time: 2019-11-18 15:26:18

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
word_dim=100
# with open(vector_path, 'r', encoding='utf8') as f:
#     word_size, word_dim = f.readline().strip().split()
#     word_dim = int(word_dim)
#     # matrix[0] for <BNK>, matrix[1] for <UNK>
#     while len(embeddings_matrix) < 2:
#         zero_vec = np.array([0 for i in range(word_dim)], dtype=np.float)
#         embeddings_matrix.append(zero_vec)
#     count = 0
#     for idx, line in enumerate(f):
#         split_data = line.strip().split()
#         if len(split_data) != (word_dim + 1):
#             logging.info('[error data] ({})\'s {}th line error data.'
#                          .format(vector_path, idx))
#             continue
#         # wrd2idx and idx2wrd, keep 0-idx for mask and 1-idx for oov
#         word = split_data[0]
#         if word not in wrd2idx:
#             wrd2idx[word] = count+2
#             idx2wrd[count+2] = word
#             count += 1
#             # vector matrix
#             vec = np.array(split_data[1:], dtype=np.float)
#             embeddings_matrix.append(vec)

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

    def Model3(self):# combine input - textcnn - flatten - dense1 # best negetive_f1: 0.78, acc: 0.892
        inputs = keras.Input(shape=(self.lens, ), )
        mask = keras.layers.Masking(mask_value=0)(inputs)
        if type(self.embeddings_matrix) != type(None):
            EMBEDDING_DIM = EMBEDDING_DIM_WORD
            embed = keras.layers.Embedding(input_dim=len(wrd2idx), output_dim=EMBEDDING_DIM, weights=[self.embeddings_matrix], trainable=False)(mask)
        else:
            EMBEDDING_DIM = EMBEDDING_DIM_CHAR
            embed = keras.layers.Embedding(input_dim=len(char2idx), output_dim=EMBEDDING_DIM)(mask)

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

    def Model4(self):# Bilstm step + flatten + dense
        inputs = keras.Input(shape=(self.lens, ), )
        mask = keras.layers.Masking(mask_value=0)(inputs)
        if type(self.embeddings_matrix) != type(None):
            EMBEDDING_DIM = EMBEDDING_DIM_WORD
            embed = keras.layers.Embedding(input_dim=len(wrd2idx), output_dim=EMBEDDING_DIM, weights=[self.embeddings_matrix], trainable=False)(mask)
        else:
            EMBEDDING_DIM = EMBEDDING_DIM_CHAR
            embed = keras.layers.Embedding(input_dim=len(char2idx), output_dim=EMBEDDING_DIM)(mask)
        rnn_layer = keras.layers.Bidirectional(keras.layers.LSTM(EMBEDDING_DIM, return_sequences=True))(embed)
        flatten_layer = keras.layers.Flatten()(rnn_layer)
        dropout = keras.layers.Dropout(0.3)(flatten_layer)
        y = keras.layers.Dense(3, activation='softmax')(dropout)
        model = keras.Model(inputs=inputs, outputs=[y])
        if self.plot:
            keras.utils.plot_model(model, to_file='model/Model4.png', show_shapes=True)
        print(model.summary())
        return model


    def Model5(self):# embed + mask + Bilstm
        inputs = keras.Input(shape=(self.lens, ), )
        mask = keras.layers.Masking(mask_value=0)(inputs)
        if type(self.embeddings_matrix) != type(None):
            EMBEDDING_DIM = EMBEDDING_DIM_WORD
            embed = keras.layers.Embedding(input_dim=len(wrd2idx), output_dim=EMBEDDING_DIM, weights=[self.embeddings_matrix], trainable=False)(mask)
        else:
            EMBEDDING_DIM = EMBEDDING_DIM_CHAR
            embed = keras.layers.Embedding(input_dim=len(char2idx), output_dim=EMBEDDING_DIM)(mask)
        rnn_layer = keras.layers.Bidirectional(keras.layers.LSTM(EMBEDDING_DIM))(embed)
        dropout = keras.layers.Dropout(0.3)(rnn_layer)
        y = keras.layers.Dense(3, activation='softmax')(dropout)
        model = keras.Model(inputs=inputs, outputs=[y])
        if self.plot:
            keras.utils.plot_model(model, to_file='model/Model5.png', show_shapes=True)
        print(model.summary())
        return model

    def Model6(self):# RCNN
        inputs = keras.Input(shape=(self.lens, ), )
        mask = keras.layers.Masking(mask_value=0)(inputs)
        if type(self.embeddings_matrix) != type(None):
            EMBEDDING_DIM = EMBEDDING_DIM_WORD
            embed = keras.layers.Embedding(input_dim=len(wrd2idx), output_dim=EMBEDDING_DIM, weights=[self.embeddings_matrix], trainable=False)(mask)
        else:
            EMBEDDING_DIM = EMBEDDING_DIM_CHAR
            embed = keras.layers.Embedding(input_dim=len(char2idx), output_dim=EMBEDDING_DIM)(mask)
        rnn_fw, rnn_bw = keras.layers.Bidirectional(keras.layers.LSTM(EMBEDDING_DIM, return_sequences=True), merge_mode=None)(embed)
        merge_layer = keras.layers.concatenate([rnn_fw, embed, rnn_bw], axis=-1)
        activation_layer = keras.layers.Activation('tanh')(merge_layer)
        pooling_layer = keras.layers.GlobalAveragePooling1D()(activation_layer)
        y = keras.layers.Dense(3, activation='softmax')(pooling_layer)
        model = keras.Model(inputs=inputs, outputs=[y])
        if self.plot:
            keras.utils.plot_model(model, to_file='model/Model6.png', show_shapes=True)
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
        earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min', restore_best_weights=True)
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
    # from sklearn.model_selection import train_test_split
    # path = '/home/tungee/桌面/jasoncheung/工信部/alg-emotion-train/data/tungee/v0_7/all_gxb.tsv'
    # df = pd.read_csv(path, sep='\t')
    # train, test = train_test_split(df, test_size=0.05)
    ###########################################
    # read data
    train_path = './data/tungee/v0_7/all_gxb_train.tsv'
    test_path = './data/tungee/v0_7/all_gxb_test.tsv'

    df_train = pd.read_csv(train_path, sep='\t').dropna()
    df_test = pd.read_csv(test_path, sep='\t').dropna()

    df_train['label'] = df_train['label'].replace('正面', 0).replace('负面', 1).replace('中性', 2)
    df_train['label'] = df_train['label'].replace('0.0', 0).replace('1.0', 1).replace('2.0', 2)
    df_test['label'] = df_test['label'].replace('正面', 0).replace('负面', 1).replace('中性', 2)
    df_test['label'] = df_test['label'].replace('0.0', 0).replace('1.0', 1).replace('2.0', 2)

    x_train = df_train['cmnt'].tolist()
    y_train = df_train['label'].tolist()
    y_train = [int(i) for i in y_train]

    x_test = df_test['cmnt'].tolist()
    y_test = df_test['label'].tolist()
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
    histories.append(('model3', utils.train_model(model2, 'model/model2-lenet5_char_v0.6.h5', nlp.Model2_1())))
    # histories.append(('model4', utils.train_model(model4, 'model/model4_char.h5', nlp.Model4())))
    # histories.append(('model5', utils.train_model(model5, 'model/model5_char.h5', nlp.Model5())))
    # histories.append(('model6', utils.train_model(model6, 'model/model6_char.h5', nlp.Model6())))
    utils.plot_history(histories, path='model/acc_char_emotion.png')
    print(utils.cal_pr(model2))
    ###########################################

    ###########################################
    # word model # 用知乎词向量还不如char
    # transfer data
    # x_train_word = [list(jieba.cut(i)) for i in x_train]
    # x_test_word = [list(jieba.cut(i)) for i in x_test]
    #
    # x_train_word = [[wrd2idx[i] if i in wrd2idx else wrd2idx['<UNK>'] for i in j] for j in x_train]
    # x_test_word = [[wrd2idx[i] if i in wrd2idx else wrd2idx['<UNK>'] for i in j] for j in x_test]
    #
    # print('padding sequences')
    # x_train_word = sequence.pad_sequences(x_train_word, maxlen=MAX_LEN, padding='post', truncating='post')
    # x_test_word = sequence.pad_sequences(x_test_word, maxlen=MAX_LEN, padding='post', truncating='post')
    #
    # print('x_train_word shape:', x_train_word.shape)
    # print('x_test_word shape:', x_test_word.shape)
    #
    # print('y_train shape:', y_train.shape)
    # print('y_test shape:', y_test.shape)
    #
    # # mlp classifier
    # clf_path = '/home/tungee/桌面/jasoncheung/工信部/alg-emotion-train/model/tungee/word2vec_mlp_word/faq_model.pkl'
    # from sklearn.neural_network import MLPClassifier
    # clf = pickle.load(open(clf_path, 'rb'))
    # x_test_clf = [list(i) for i in x_test_word]
    # x_test_clf = [[embeddings_matrix[i] for i in j] for j in x_test_clf]
    # x_test_clf = [sum(i) for i in x_test_clf]
    # x_test_clf = np.array(x_test_clf)
    # pred = clf.predict(x_test_clf)
    # true = list(np.argmax(y_test, axis=1))
    # report = classification_report(true, pred)
    #
    # # model
    # nlp = nlpModel(MAX_LEN,  embeddings_matrix, plot=False)
    # model1 = nlp.Model1()
    # model2 = nlp.Model2()
    # model3 = nlp.Model3()
    # model4 = nlp.Model4()
    # model5 = nlp.Model5()
    # model6 = nlp.Model6()
    #
    # utils = jason_tools(x_train_word, y_train, x_test_word, y_test)
    # histories = []
    # histories.append(('model1', utils.train_model(model1, 'model/model1_word.h5', nlp.Model1())))
    # histories.append(('model2', utils.train_model(model2, 'model/model2_word.h5', nlp.Model2())))
    # histories.append(('model3', utils.train_model(model3, 'model/model3_word.h5', nlp.Model3())))
    # histories.append(('model4', utils.train_model(model4, 'model/model4_word.h5', nlp.Model4())))
    # histories.append(('model5', utils.train_model(model5, 'model/model5_word.h5', nlp.Model5())))
    # histories.append(('model6', utils.train_model(model6, 'model/model6_word.h5', nlp.Model6())))
    # utils.plot_history(histories, path='model/acc_word_emotion.png')
    ###########################################

    # #############
    # # check data
    # data_path = './data/tungee/v0_7/Train_gxb_v0.3.tsv'
    # df = pd.read_csv(data_path, sep='\t', header=None)
    #
    # df.columns = ['id', 'label', 'cmnt']
    # df['label'] = df['label'].replace('正面', 0).replace('负面', 1).replace('中性', 2)
    # cmnt = df['cmnt'].tolist()
    # cmnt = [i.replace('\n', '').replace('\t', '').replace(' ', '').replace('　', '').strip() for i in cmnt]
    # df['cmnt'] = cmnt
    #
    # print('中性: ', len(df.loc[df['label'] == 2]))
    # print('正面: ', len(df.loc[df['label'] == 0]))
    # print('负面: ', len(df.loc[df['label'] == 1]))
    #
    # df_train, df_test = train_test_split(df, test_size=0.05, random_state=42)
    #
    # print('df_train')
    # print('中性: ', len(df_train.loc[df_train['label'] == 2]))
    # print('正面: ', len(df_train.loc[df_train['label'] == 0]))
    # print('负面: ', len(df_train.loc[df_train['label'] == 1]))
    # print('df_test')
    # print('中性: ', len(df_test.loc[df_test['label'] == 2]))
    # print('正面: ', len(df_test.loc[df_test['label'] == 0]))
    # print('负面: ', len(df_test.loc[df_test['label'] == 1]))
    #
    # df_train.to_csv('./data/tungee/v0_7/all_gxb_train.tsv', sep='\t', index=False)
    # df_test.to_csv('./data/tungee/v0_7/all_gxb_test.tsv', sep='\t', index=False)
    # #############



