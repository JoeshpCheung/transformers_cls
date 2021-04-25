#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    ~~~~~~~~~~~~~~~~~~~~~~~
    :author: JoeshpCheung
    :python version: 3.6
"""
from torch.utils.data.dataset import Dataset
import numpy as np
import pandas as pd
import json
import librosa

class DataLoader(object):
    def __init__(self, ignore_index=-100, max_length=512):
        self.ignore_index = ignore_index
    
    def get_mfcc_feature(self, data, sr):
        tmp_feature = librosa.feature.mfcc(data, sr=sr, n_mfcc=20)  # 默认n_mfcc=20
        # mel = librosa.feature.melspectrogram(data, sr=sr, n_fft=1024, n_mels=256)  # 计算Mel scaled 频谱
        # mel = librosa.power_to_db(mel).T  #  功率转dB
        return tmp_feature
        
    def load_voice(self, fp):
        # 下载音频数据
        # 先只考虑mfcc
        data, sr = librosa.load(path=fp, sr=None)

        return data, sr
    
    def cut_voice(self, data, sr, hit_time, duration=5):
        start_time = int((hit_time - duration) * sr)
        end_time = int(hit_time * sr)
        tmp_data = data[start_time : end_time]
        return tmp_data
        
    def load_datas(self, data_dir, is_train: bool = True):
        if is_train:
            tmp_file = data_dir + '/wrap_output.txt'
            print('fp: ', tmp_file)
            id_ = []
            text = []
            label = []
            hit_time = []
            with open(tmp_file, 'r') as fr:
                for idx, i in enumerate(fr):
                    tmp = json.loads(i)
                    id_.append(tmp.get('_id'))
                    text.append(tmp.get('asr_text'))
                    label.append(tmp.get('call_status'))
                    hit_time.append(float(tmp.get('detect_result_hit_duration')) / 1000)
        mfcc = []
        mfcc_target = []  # 命中关键词前5秒录音
        duration = []
        for idx, i in enumerate(id_):
            tmp_file = data_dir + '/slice_media_dir/' + str(i) + '.wav'

            tmp_data, tmp_sr = self.load_voice(tmp_file)
            tmp_mfcc = self.get_mfcc_feature(tmp_data, tmp_sr)
            tmp_duration = len(tmp_data) / tmp_sr
            
            duration.append(tmp_duration)
            mfcc.append(tmp_mfcc)
            '''
            # 获得击中及前5秒特征数据
            print(hit_time[idx])
            tmp_data = self.cut_voice(tmp_data, tmp_sr, hit_time[idx], duration=5)
            print(tmp_data.shape, tmp_sr)
            tmp_mfcc_hit = self.get_mfcc_feature(tmp_data, tmp_sr)
            
            
            mfcc_target.append(tmp_mfcc_hit)
            '''
            
        df = pd.DataFrame(data={'_id': id_, 'text': text, 'label': label, 'hit_time': hit_time, 'features': mfcc})

        return df

class Dataset_end(Dataset):
    def __init__(self, df, num_classes):
        super(Dataset_end, self).__init__()
        self.datas = df.features_padding.tolist()
        self.labels = df.true.tolist()
        self.num_classes = num_classes
        # self.ld = {5: 0, 9:1, 16:2}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # label = self.ld[self.labels[index]]
        tmp_label = self.labels[index]
        label = [0 for i in range(self.num_classes)]
        for idx in tmp_label:
            label[idx] = 1

        data = self.datas[index]

        data = np.array(data)
        return (data.astype(np.int64), label)



