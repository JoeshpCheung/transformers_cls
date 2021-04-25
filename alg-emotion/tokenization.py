#!/usr/bin/env python3
# coding=utf-8

"""    
    @File: tokenization.py
    @Desc: 基于jieba，自己的分词器
    @Author: Chris Tao
    @Date Created: 2019/4/11
"""

import jieba.posseg as psg

USELESS_POS = {'x'}
STOPWORD_FILE = None


class ChiWordTokenizer(object):
    def __init__(self,
                 useless_pos=USELESS_POS,
                 stopwords_file=STOPWORD_FILE):
        self.useless_pos = useless_pos
        self.stopwords = self._get_stopwords(stopwords_file) if stopwords_file else {}
        self.vocab, self.id2vocab, self.vocab_size = dict(), dict(), 0
        self.documents = []

    @staticmethod
    def _get_stopwords(stopwords_file):
        fin = open(stopwords_file, 'r', encoding='utf8')
        stopwords = set([line.strip() for line in fin])
        fin.close()
        return stopwords

    def feed(self, text):
        self.documents.append(text)

    def bulid_vocab(self):
        for text in self.documents:
            tokens = []    # 符合要求的token列表
            for item in psg.cut(text):
                if item.flag in self.useless_pos:
                    continue
                if item.word in self.stopwords:
                    continue
                tokens.append(item.word)
            self._add_tokens_to_vocab(tokens)

        self.id2vocab = {w_idx: w for w, w_idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def update_vocab(self, choose_token_ids):
        new_vocab = dict()
        for idx, token_id in enumerate(choose_token_ids):
            token = self.id2vocab[token_id]
            new_vocab[token] = idx
        self.vocab = new_vocab.copy()
        self.id2vocab = {w_idx:w for w, w_idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def _add_tokens_to_vocab(self, tokens):
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

    def tokenize(self, text, oov=False):
        split_token = []
        for item in psg.cut(text):
            if item.flag in self.useless_pos:
                continue
            if item.word in self.stopwords:
                continue
            if not oov and item.word not in self.vocab:
                continue
            split_token.append(item.word)
        return split_token

    def convert_tokens_to_ids(self, tokens):
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])
        return ids

    def convert_ids_to_tokens(self, ids):
        tokens = []
        for _id in ids:
            tokens.append(self.id2vocab[_id])
        return tokens


class ChiMixinTokenizer(object):
    def __init__(self,
                 useless_pos={'x'},
                 stopwords_file=None):
        self.useless_pos = useless_pos
        self.stopwords = self._get_stopwords(stopwords_file) if stopwords_file else {}
        self.vocab, self.id2vocab, self.vocab_size = dict(), dict(), 0
        self.documents = []

    @staticmethod
    def _get_stopwords(stopwords_file):
        fin = open(stopwords_file, 'r', encoding='utf8')
        stopwords = set([line.strip() for line in fin])
        fin.close()
        return stopwords

    def feed(self, text):
        self.documents.append(text)

    def bulid_vocab(self):
        for text in self.documents:
            tokens = []    # 符合要求的token列表
            for item in psg.cut(text):
                if item.flag in self.useless_pos:
                    continue
                if item.word in self.stopwords:
                    continue
                tokens.append(item.word)
            self._add_tokens_to_vocab(tokens)

        self.id2vocab = {w_idx: w for w, w_idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def update_vocab(self, choose_token_ids):
        new_vocab = dict()
        for idx, token_id in enumerate(choose_token_ids):
            token = self.id2vocab[token_id]
            new_vocab[token] = idx
        self.vocab = new_vocab.copy()
        self.id2vocab = {w_idx: w for w, w_idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def _add_tokens_to_vocab(self, tokens):
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

        co_tokens = self._get_co_tokens(tokens)
        for co_token in co_tokens:
            if co_token not in self.vocab:
                self.vocab[co_token] = len(self.vocab)

    @staticmethod
    def _get_co_tokens(tokens):
        co_tokens = []
        tokens = list(set(tokens))
        N = len(tokens)
        if N < 2:
            return co_tokens
        tokens.sort()
        for i in range(N - 1):
            for j in range(i, N):
                token1 = tokens[i]
                token2 = tokens[j]
                co_tokens.append(token1 + '_' + token2)
        return co_tokens

    def tokenize(self, text, oov=False):
        split_token = []
        tokens = []
        for item in psg.cut(text):
            if item.flag in self.useless_pos:
                continue
            if item.word in self.stopwords:
                continue
            tokens.append(item.word)

        for token in tokens:
            if not oov and token not in self.vocab:
                continue
            split_token.append(token)

        co_tokens = self._get_co_tokens(tokens)
        for co_token in co_tokens:
            if not oov and co_token not in self.vocab:
                continue
            split_token.append(co_token)
        return split_token

    def convert_tokens_to_ids(self, tokens):
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])
        return ids

    def convert_ids_to_tokens(self, ids):
        tokens = []
        for _id in ids:
            tokens.append(self.id2vocab[_id])
        return tokens