#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    ~~~~~~~~~~~~~~~~~~~~~~~
    :author: JoeshpCheung
    :python version: 3.6
"""
import numpy as np
import json

def matrix_make_up(audio, h=20, l=1500):
    new_audio = []
    for adx, aa in enumerate(audio):
        zeros_matrix = np.zeros([h, l], np.int8)
        a, b = np.array(aa).shape
        
        if a > h:
            a = h
        if b > l:
            b = l
            
        for i in range(a):
            for j in range(b):
                zeros_matrix[i, j] = zeros_matrix[i, j] + aa[i, j]
        new_audio.append(zeros_matrix)
    return new_audio


def read_json(path):
    res = []
    with open(path, 'r') as fr:
        for idx, i in enumerate(fr):
            res.append(json.loads(i))

    return res
