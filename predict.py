#!/usr/bin/python
# -*- coding:utf8 -*-
# @Author : tuolifeng
# @Time : 2018/7/2 上午10:03
# @File : predict.py
# @Software : IntelliJ IDEA
# @Email ： 909709223@qq.com
# 结果预测
from sklearn.externals import joblib
import pandas as pd

svm_model = joblib.load("model_dga_classifier_20180630.model")
raw_test_dga = pd.read_csv('domain_vectorized_features.txt')
X_test_dga=raw_test_dga.loc[:,'bi_rank':'vowel_ratio'].values
probas_test_dga_score = pd.DataFrame(pd.DataFrame(svm_model.predict_proba(X_test_dga))[1])
dga_domain = pd.DataFrame(raw_test_dga.loc[:,'ip'].values)
rst_score_dga_df = pd.concat([dga_domain, probas_test_dga_score],axis=1)
rst_score_dga_df.to_csv("predict_rslt.csv")