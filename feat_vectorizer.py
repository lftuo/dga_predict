#!/usr/bin/python
# -*- coding:utf8 -*-
# @Author : tuolifeng
# @Time : 2018/6/30 下午12:02
# @File : feat_vectorizer.py.py
# @Software : IntelliJ IDEA
# @Email ： 909709223@qq.com

'''
this script translate features to training table
特征向量化DictVectorizer()
'''
from collections import Counter
from operator import itemgetter
from sklearn.externals import joblib

fi = open('features_norm.txt','r')
header = fi.readline().strip().split(',')
header_dict = dict((j,i) for i,j in enumerate(header))
feat_table = list()
for f in fi:
    # domain,cla,tld,entropy,f_len,norm_entropy,bigram,trigram=f.rstrip('\n').split('\t')
    ll = f.rstrip('\n').split(',')
    # dummy,domain,cla,tld,entropy,f_len,norm_entropy,vowel_ratio,uni_rank,bi_rank,tri_rank,uni_std,bi_std,tri_std=f.rstrip('\n').split(',')
    feat_dict = dict()
    feature_header = header
    for f in feature_header:
        if f in ['','ip','private_tld']: continue
        feat = ll[header_dict[f]]
        if not f=='tld': feat=float(feat)
        else: feat=feat.lower()

        feat_dict[f]=feat
    domain = itemgetter(1)(ll)
    '''
    feat_dict['tld']=tld.lower()
    feat_dict['entropy']=float(entropy)
    feat_dict['len']=float(f_len)
    feat_dict['norm_entropy']=float(norm_entropy)
    feat_dict['uni_rank']=float(uni_rank)
    feat_dict['bi_rank']=float(bi_rank)
    feat_dict['tri_rank']=float(tri_rank)
    feat_dict['uni_std']=float(uni_std)
    feat_dict['bi_std']=float(bi_std)
    feat_dict['tri_std']=float(tri_std)
    feat_dict['vowel_ratio']=float(vowel_ratio)
    '''
    '''
    bigram=Counter(bigram.split(',')).most_common()
    trigram=Counter(trigram.split(',')).most_common()
    for b,freq in bigram:
        if not b =='':
            feat_dict['bigram_'+b]=float(freq)
    for t,freq in trigram:
        if not t =='':
            feat_dict['trigram_'+b]=float(freq)
    '''
    feat_table.append([domain,feat_dict])

fi.close()

measurements = [feat_dict for domain,feat_dict in feat_table]

vec = joblib.load("dict_vectorizer_20180630.model")
feature_list = vec.transform(measurements).toarray()
feature_header = vec.get_feature_names()

fw_out = open('domain_vectorized_features.txt','w')

fw_out.write('ip,%s\n'%(','.join(feature_header)))

ground_truth = [domain for domain,feat_dict in feat_table]
for domain,feats in zip(ground_truth,feature_list):
    joined_feats = ','.join('%.2f'%i for i in feats)
    fw_out.write('%s,%s\n'%(domain,joined_feats))

fw_out.close()
