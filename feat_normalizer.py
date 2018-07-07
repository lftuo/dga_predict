#!/usr/bin/python
# -*- coding:utf8 -*-
# @Author : tuolifeng
# @Time : 2018/6/30 上午11:53
# @File : feat_normalizer.py.py
# @Software : IntelliJ IDEA
# @Email ： 909709223@qq.com

import pandas as pd
import numpy as np
black_list = ['ip','tld']

feat_table = pd.read_csv('features.txt',delimiter='\t')

header = list(feat_table.columns)
feat_matrix = pd.DataFrame()
for i in header:
    if i in black_list:
        feat_matrix[i]=feat_table.ix[:,i]
    else:
        line = feat_table.ix[:,i]
        mean_ = line.mean()
        max_ = line.max()
        min_ = line.min()
        feat_matrix[i]=(line-mean_)/(max_-min_)
    print('converted %s'%i)

feat_matrix.to_csv('features_norm.txt')