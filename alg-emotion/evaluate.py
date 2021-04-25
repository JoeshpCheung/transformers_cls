#!/usr/bin/env python3
# coding=utf-8

"""
    @File: evaluate.py
    @Desc: 统一的评估函数
    @Author: Chris Tao
    @Date Created: 2019/4/11
"""

import logging

import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s:%(levelname)s:%(message)s")


def evaluate(pred_file, eval_file, belta=0.5):
    data = pd.read_csv(pred_file,
                       sep='\t',
                       names=['_id', 'label', 'text', 'prob', 'pred'],
                       encoding='utf8')

    fout = open(eval_file, 'w', encoding='utf8')
    fout.write('[{}] acc evaluation is:\n'.format(pred_file))
    fout.write('-' * 80 + '\n')
    fout.write('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n'
               .format(' Thres', ' Total', ' Right', '  ACC ', '  TP  ', '  FP  ',
                       '  FN  ', '   P  ', '   R  ', '   F  '))
    best_acc_thres, best_ACC = 0, 0
    best_f_thres, best_P, best_R, best_F = 0, 0, 0, 0
    total_cnt = len(data)
    for i in range(0, 100, 5):
        thres = float(i) / 100
        data_cp = data.copy()
        data_cp.loc[data_cp['prob'] < thres, 'pred'] = 'OTHER'
        right_cnt = len(data_cp.loc[data_cp['label'] == data_cp['pred']])
        ACC = round(float(right_cnt) / total_cnt, 4)
        if best_ACC < ACC:
            best_ACC, best_acc_thres = ACC, thres

        TP = len(data_cp.loc[(data_cp['pred'] == data_cp['label']) &
                             (data_cp['pred'] != 'OTHER')])
        FP = len(data_cp.loc[(data_cp['pred'] != data_cp['label']) &
                             (data_cp['pred'] != 'OTHER')])
        FN = len(data_cp.loc[(data_cp['pred'] != data_cp['label']) &
                             (data_cp['label'] != 'OTHER')])
        P, R, F = cal_f(TP, FP, FN, belta)
        if best_F < F:
            best_F, best_f_thres, best_P, best_R = F, thres, P, R

        fout.write('{:6.2f}, {:6d}, {:6d}, {:.4f}, {:6d}, {:6d}, {:6d}, {:.4f}, {:.4f}, {:.4f}\n'
                   .format(thres, total_cnt, right_cnt, ACC, TP, FP, FN, P, R, F))
    fout.write('-' * 80 + '\n')
    fout.write('Best [Thres, ACC] is [{:6.2f}, {:.4f}]\n'
               .format(best_acc_thres, best_ACC))
    fout.write('Best [Thres, P, R, F{}] is [{:6.2f}, {:.4f}, {:.4f}, {:.4f}]\n'
               .format(belta, best_f_thres, best_P, best_R, best_F))

    logging.info('[{}] evalution done.'.format(pred_file))


def cal_f(TP, FP, FN, belta=1.0):
    P = float(TP) / (TP + FP) if TP > 0 else 0
    R = float(TP) / (TP + FN) if TP > 0 else 0
    F = (1.0 + belta ** 2) * P * R / ((belta ** 2) * P + R) \
        if (P * R) > 0 else 0
    return P, R, F


def acc_err_detail(pred_file, eval_file, acc_err_file):
    best_thres = 0
    fin = open(eval_file, 'r', encoding='utf8')
    for line in fin:
        if line.startswith('Best [Thres, ACC]'):
            tmp = line[line.index('[') + 1:]
            tmp = tmp[tmp.index('[') + 1:].strip().replace(']', '')
            thres, _ = tmp.split(',')
            best_thres = float(thres)
            break
    logging.info('[{}] found best thres: {}'.format(eval_file, best_thres))

    data = pd.read_csv(pred_file,
                       sep='\t',
                       names=['_id', 'label', 'text', 'prob', 'pred'],
                       encoding='utf8')
    data.loc[data['prob'] < best_thres, 'pred'] = '同QA无关'
    data.loc[data['label']!=data['pred']].to_excel(acc_err_file, index=False)
    logging.info('ACC Error detail is [{}]'.format(acc_err_file))



def f_err_detail():
    pass


def demo():
    pred_0_file = './model/tungee/mybert/pred_0.tsv'
    eval_0_file = './model/tungee/mybert/eval_0.txt'
    pred_1_file = './model/tungee/mybert/pred_1c.tsv'
    eval_1_file = './model/tungee/mybert/eval_1c.txt'
    pred_2_file = './model/tungee/mybert/pred_2e.tsv'
    eval_2_file = './model/tungee/mybert/eval_2e.txt'
    acc_err_detail_1_file = './model/tungee/mybert/acc_err_1.xlsx'
    acc_err_detail_2_file = './model/tungee/mybert/acc_err_2b.xlsx'
    # evaluate(pred_0_file, eval_0_file)
    # evaluate(pred_1_file, eval_1_file)
    # evaluate(pred_2_file, eval_2_file)

    pred_3_file = './model/tungee/qiyu/qiyu_pred_20190531.tsv'
    eval_3_file = './model/tungee/qiyu/qiyu_eval_20190531.txt'
    evaluate(pred_3_file, eval_3_file)

    # acc_err_detail(pred_2_file, eval_2_file, acc_err_detail_2_file)


if __name__ == '__main__':
    demo()