# *-* coding:utf-8 *-*
"""
    Description:
    ~~~~~~~~~~~~~~~~~~~~~~~
    @project: alg-emotion-train
    @author: JasonCheung
    @file: check_data.py
    @editor: PyCharm
    @time: 2019-11-18 16:41:56

"""
import pandas as pd
from sklearn.metrics import classification_report

check_path = '/home/tungee/桌面/jasoncheung/工信部/alg-emotion-train/model/tungee/word2vec_mlp_word/pred_v0_7.tsv'
df = pd.read_csv(check_path, sep='\t')
pred = df['label'].tolist()

true_path = '/home/tungee/桌面/jasoncheung/工信部/alg-emotion-train/data/tungee/v0_7/all_gxb_test.tsv'
df = pd.read_csv(true_path, sep='\t')
true = df['label'].tolist()
wrong = (df['auto_label']-df['label']).tolist()
wrong = [1 if i != 0 else 0 for i in wrong]
df['wrong'] = wrong
df.to_excel(check_path, index=False)
target_names = ['负面', '中性']
print(classification_report(true, pred, target_names=target_names))
