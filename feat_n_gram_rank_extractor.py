#!/usr/bin/python
# -*- coding:utf8 -*-
# @Author : tuolifeng
# @Time : 2018/6/30 上午11:08
# @File : feat_n_gram_rank_extractor.py.py
# @Software : IntelliJ IDEA
# @Email ： 909709223@qq.com
from tldextract import tldextract

def ave(list_):
    if len(list_)==0:
        return 0
    else:
        return sum(list_)/float(len(list_))

def bigrams(words):
    wprev = None
    for w in words:
        if not wprev==None:
            yield (wprev, w)
        wprev = w

def trigrams(words):
    wprev1 = None
    wprev2 = None
    for w in words:
        if not (wprev1==None or wprev2==None):
            yield (wprev1,wprev2, w)
        wprev1 = wprev2
        wprev2 = w

private_tld_file = open('private_tld.txt','r')
private_tld = set(f.strip() for f in private_tld_file)#black list for private tld
private_tld_file.close()

# training_w_tld.txt是conficker_alexa_training.txt解析顶级域名后的文件
fi = open('training_w_tld.txt','r')
fw = open('gram_ranks_training.txt','w')
fw.write('domain,s1,s2,s3,core\n')

# n_gram_rank_freq.txt是top-100k.csv(alexa_100k.txt)获取unigram/bigram/trigram的基准排名后的文件
n_gram_file = open('n_gram_rank_freq.txt','r')
gram_rank_dict = dict()
for i in n_gram_file:
    cat,gram,freq,rank = i.strip().split(',')
    gram_rank_dict[gram]=int(rank)
n_gram_file.close()

for f in fi:
    domain, tld = f.strip().split('\t')
    strip_domain = domain
    ext = tldextract.extract(strip_domain)
    if len(ext.domain)>4 and ext.domain[:4]=='xn--':
        continue
    # add begin and end
    main_domain = '$'+ext.domain+'$'
    tld = ext.suffix
    # check if it is a private tld
    if tld in private_tld:
        # quick hack: if private tld, use its last part of top TLD
        tld_list = tld.split('.')
        tld = tld_list[-1]
        # and overwrite the main domain
        main_domain = '$'+tld_list[-2]+'$'
    unigram_rank = [gram_rank_dict[i] if i in gram_rank_dict else 0 for i in main_domain[1:-1]]
    # extract the bigram
    bigram_rank = [gram_rank_dict[''.join(i)] if ''.join(i) in gram_rank_dict else 0 for i in bigrams(main_domain)]
    # extract the bigram
    trigram_rank = [gram_rank_dict[''.join(i)] if ''.join(i) in gram_rank_dict else 0 for i in trigrams(main_domain)]

    try:
        fw.write('%s,%.2f,%.2f,%.2f,%s\n'%(domain,ave(unigram_rank),ave(bigram_rank),ave(trigram_rank),main_domain))
    except UnicodeEncodeError:
        # some unicode problem for some strange domains
        continue

fw.close()
fi.close()
