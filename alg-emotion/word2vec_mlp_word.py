#!/usr/bin/env python3
# coding=utf-8

"""    
    @File: word2vec_mlp_word.py
    @Desc: 基于word2vec,再构建mlp的模型
    @Author: Chris Tao
    @Date Created: 2019/05/05
"""

import logging
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier

from evaluate import evaluate
from tokenization import ChiWordTokenizer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s:%(levelname)s:%(message)s")


class Word2Vec(object):
    def __init__(self, word2vec_file):
        self.w2v = dict()
        self.word_dim = 0
        self._load_word2vec(word2vec_file)

    def _load_word2vec(self, word2vec_file):
        logging.info('[load word2vec] begin.')
        fin = open(word2vec_file, 'r', encoding='utf8')
        word_size, word_dim = fin.readline().strip().split()
        self.word_dim = int(word_dim)
        for idx, line in enumerate(fin):
            split_data = line.strip().split()
            if len(split_data) != (self.word_dim + 1):
                logging.info('[error data] ({})\'s {}th line error data.'
                             .format(word2vec_file, idx))
                continue
            word = split_data[0]
            vec = np.array(split_data[1:], dtype=np.float)
            self.w2v[word] = vec
        fin.close()
        logging.info('[load word2vec] done.')


class Word2VecMLPWord(object):
    def __init__(self, faq_file, label_file, word2vec_file, model_pkl_file):
        self.tokenization = ChiWordTokenizer()
        self.word2vec = Word2Vec(word2vec_file)
        self.model_pkl_file = model_pkl_file
        self.faq_file = faq_file
        self.label_file = label_file
        self.label2id, self.id2label = self._get_labels()
        self.faqs, self.labels = self._load_data()
        self.X_data, self.y_label = self._cal_data_vec()
        self.clf = self._fit_model()
        self._save_model()
        return self.X_data, self.y_label
        # if os.path.exists(self.model_pkl_file):
        #     self._load_model()
        # else:
        #     self.faq_file = faq_file
        #     self.label_file = label_file
        #     self.label2id, self.id2label = self._get_labels()
        #     self.faqs, self.labels = self._load_data()
        #     self.X_data, self.y_label = self._cal_data_vec()
        #     self.clf = self._fit_model()
        #     self._save_model()

    def _get_labels(self):
        label2id, id2label = dict(), dict()
        with open(self.label_file, 'r', encoding='utf8') as fin:
            for line in fin:
                _id, label = line.strip().split('\t')
                label2id[label], id2label[int(_id)] = int(_id), label
        return label2id, id2label

    def _load_data(self):
        N = 0
        with open(self.faq_file, 'r', encoding='utf8') as fin:
            for line in fin:
                N += 1
                # print(N, line.strip().split('\t'))
                try:
                    _, _, text = line.strip().split('\t')
                    self.tokenization.feed(text)
                except:
                    pass
        self.tokenization.bulid_vocab()

        # faqs, labels = [], []
        # with open(self.faq_file, 'r', encoding='utf8') as fin:
        #     for idx, line in enumerate(fin):
        #         try:
        #             _, label, text = line.strip().split('\t')
        #             faqs.append(text)
        #             labels.append(label)
        #         except:
        #             pass
        df = pd.read_csv(self.faq_file, sep='\t').dropna()
        faqs = np.array(df['cmnt'].tolist())
        labels = np.array(df['label'].tolist())

        logging.info('[load data] done.')
        return faqs, labels

    def _cal_text_word2vec(self, text):
        vec = np.zeros([ self.word2vec.word_dim], dtype=np.float)
        for token in self.tokenization.tokenize(text, oov=False):
            if token in self.word2vec.w2v:
                vec += self.word2vec.w2v[token]
        return vec

    def _cal_data_vec(self):
        N = len(self.faqs)
        nonzero_idx = []  # 分类器只用非全零的向量
        X_data = np.zeros([N, self.word2vec.word_dim], dtype=np.float)
        for idx, faq in enumerate(self.faqs):
            w2v = self._cal_text_word2vec(faq)
            if np.any(w2v):
                X_data[idx] = w2v
                nonzero_idx.append(idx)
        y_label = np.zeros(N)
        print(y_label)
        for idx, label in enumerate(self.labels):
            y_label[idx] = self.label2id[label]

        X_data = X_data[nonzero_idx]
        y_label = y_label[nonzero_idx]

        # X_data = np.zeros([N, self.word2vec.word_dim], dtype=np.float)
        # for idx, faq in enumerate(self.faqs):
        #     X_data[idx] += self._cal_text_word2vec(faq)
        # y_label = np.zeros([N])
        # for idx, label in enumerate(self.labels):
        #     y_label[idx] = self.label2id[label]
        # logging.info('[cal data vec] done.')
        return X_data, y_label

    def _fit_model(self):
        clf = MLPClassifier(hidden_layer_sizes=100,
                            activation='tanh',
                            solver='adam',
                            alpha=2e-4,
                            learning_rate_init=1e-4,
                            max_iter=2000,
                            shuffle=True,
                            random_state=502)
        clf.fit(self.X_data, self.y_label)
        logging.info('[model fit] done.')
        return clf

    def init_model(self, test_file):
        self.clf = MLPClassifier(hidden_layer_sizes=100,
                            activation='tanh',
                            solver='adam',
                            alpha=2e-4,
                            learning_rate_init=1e-4,
                            max_iter=2000,
                            random_state=502)
        docs = []
        with open(test_file, 'r', encoding='utf8') as fin:
            for line in fin:
                _id, label, text = line.strip().split('\t')
                docs.append(text)
        self.X_test = np.zeros([len(docs), self.word2vec.word_dim])
        for idx, text in enumerate(docs):
            self.X_test[idx] = self._cal_text_word2vec(text)

    def _save_model(self):
        fout = open(self.model_pkl_file, 'wb')
        pickle.dump(self.clf, fout, protocol=3)
        fout.close()
        logging.info('[save model to {}] done.'.format(self.model_pkl_file))

    def _load_model(self):
        logging.info('[load model from {}] done.'.format(self.model_pkl_file))

    def feed_prob(self, text):
        vec = self._cal_text_word2vec(text).reshape([1, -1])
        if np.any(vec):
            sim = self.clf.predict_proba(vec).flatten()
        else:
            sim = np.ones(len(self.label2id)) / len(self.label2id)
        best_label_id = sim.argmax()
        result = {'result': {'prob': round(sim.max(), 4),
                             'label': self.id2label[best_label_id]
                             }
                  }
        return result

    def feed(self, text):
        vec = self._cal_text_word2vec(text).reshape([1, -1])
        label_id = int(self.clf.predict(vec)[0])
        result = {'result': {'prob': 1.0,
                             'label': self.id2label[label_id]
                             }
                  }
        return result


def batch_feed(model, test_file, pred_file):
    logging.info('[{}] pred begin.'.format(test_file))
    fout = open(pred_file, 'w', encoding='utf8')
    with open(test_file, 'r', encoding='utf8') as fin:
        for idx, line in enumerate(fin):
            if idx % 1000 == 0:
                logging.info('[{}th] done.'.format(idx))
            _id, label, text = line.strip().split('\t')
            result = model.feed_prob(text)
            pred_prob = str(result['result']['prob'])
            pred_label = result['result']['label']
            new_line = '\t'.join([_id, label, text, pred_prob, pred_label])
            fout.write(new_line + '\n')
    fout.close()
    logging.info('[{}] pred done, write result to [{}]'
                 .format(test_file, pred_file))


def demo():
    faq_file = './data/tungee/v0_7/all_gxb_train.tsv'
    label_file = './data/tungee/v0_7/label_1.tsv'
    word2vec_file = './word2vec/sgns.zhihu.word'
    model_pkl_file = './model/tungee/word2vec_mlp_word/faq_model.pkl'
    X_data, y_label = Word2VecMLPWord(faq_file=faq_file,
                            label_file=label_file,
                            word2vec_file=word2vec_file,
                            model_pkl_file=model_pkl_file)
    clf = model.clf
    # logging.info('{}'.format(model.feed_prob('你们是怎么定义提供给我们的线索是精准的呢？')))
    # test_file = './data/tungee/v0_7/test.tsv'
    test_file = './data/tungee/v0_7/all_gxb_test.tsv'
    # pred_file = './model/tungee/word2vec_mlp_word/pred_v0_7.tsv'
    pred_file = './model/tungee/word2vec_mlp_word/pred_v0_7.tsv'
    # eval_file = './model/tungee/word2vec_mlp_word/eval_v0_7.txt'
    eval_file = './model/tungee/word2vec_mlp_word/eval_v0_7.txt'
    batch_feed(model, test_file, pred_file)
    evaluate(pred_file, eval_file)
    return


if __name__ == '__main__':

    # #############
    # check data
    import pandas as pd

    data_path = './data/tungee/v0_7/Train_gxb_v0.3.tsv'
    df = pd.read_csv(data_path, sep = '\t').dropna()
    print('中性: ', len(df.loc[df['label'] == '中性']))
    print('正面: ', len(df.loc[df['label'] == '正面']))
    print('负面: ', len(df.loc[df['label'] == '负面']))

    from sklearn.model_selection import train_test_split
    df_train, df_test = train_test_split(df, test_size=0.05, random_state=42)

    print('df_train')
    print('中性: ', len(df_train.loc[df_train['label'] == '中性']))
    print('正面: ', len(df_train.loc[df_train['label'] == '正面']))
    print('负面: ', len(df_train.loc[df_train['label'] == '负面']))
    print('df_test')
    print('中性: ', len(df_test.loc[df_test['label'] == '中性']))
    print('正面: ', len(df_test.loc[df_test['label'] == '正面']))
    print('负面: ', len(df_test.loc[df_test['label'] == '负面']))

    df_train.to_csv('./data/tungee/v0_7/all_gxb_train.tsv', sep='\t', index=False)
    df_test.to_csv('./data/tungee/v0_7/all_gxb_test.tsv', sep='\t', index=False)
    # #############
    demo()