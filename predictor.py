#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/8/1 15:19
# @Author  : fhh
# @FileName: predictor.py
# @Software: PyCharm
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from feature import *
import torch
from pathlib import Path
import argparse
import joblib


def ArgsGet():
    parse = argparse.ArgumentParser(description='pLM4PEP')
    parse.add_argument('-file', type=str, default='./test.fasta', help='fasta file')
    parse.add_argument('-out_path', type=str, default='./result', help='output path')
    args = parse.parse_args()
    return args


def predict_indep(features):
    # print(features.shape)
    # print(alldata.keys())
    model1 = joblib.load('./model.joblib')  # 读取之前保存的模型
    scaling = StandardScaler()      # 读取之前保存的归一化方法
    # f_scaling.fit(features)
    x_test = scaling.transform(features)
    y_pred = model1.predict(x_test)
    return y_pred


def pre_my(data, output_path, names):
    result = predict_indep(data)
    for i in range(len(result)):
        with open(output_path, 'a') as f:
            f.write(names[i])
            f.write('\n')
            if result[i] == 1:
                f.write('positive')
                f.write('\n')
            else:
                f.write('negative')
                f.write('\n')
            f.close()


def ReadPeptides(file_path):  # 读取肽序列文件
    posdata = open(file_path, 'r')
    pos = []
    names = []
    for l in posdata.readlines():
        if l[0] == '>':
            names.append(l)
            continue
        else:
            pos.append(l.strip('\t0\n'))
    posdata.close()
    return pos, names


def CombineFeature(pep, featurelist):  #把featurelist里包含的feature堆叠起来
    a = np.empty([len(pep), 1]) #初始化特征矩阵，行数为训练集样本数，列数为1
    fname = []  #创建空列表保存特征名称
    vocab_name = []
    # print(a)
    if 'bertfea' in featurelist:
        # 加载预训练好的esm2模型
        f_bertfea = torch.load('./esm2/train_dataset_500.pt')  # 这里改为调用ESM2模型提取特征的代码

        b = MinMaxScaler().fit_transform(f_bertfea)
        a = np.column_stack((a, b))
        fname = fname + ['bertfea'] * b.shape[1]
        print(b.shape)

    return a[:, 1:], fname, vocab_name


if __name__ == '__main__':
    args = ArgsGet()
    file = args.file  # fasta file
    output_path = args.out_path  # output path

    # building output path directory
    Path(output_path).mkdir(exist_ok=True)

    # read fasta file
    pep, names = ReadPeptides(file)
    all_feature = ['bertfea']

    # extract features
    features, _, _ = CombineFeature(pep, all_feature)

    # prediction
    pre_my(features, names, output_path)
