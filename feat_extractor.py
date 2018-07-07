#!/usr/bin/python
# -*- coding:utf8 -*-
# @Author : tuolifeng
# @Time : 2018/6/30 上午11:27
# @File : feat_extractor.py.py
# @Software : IntelliJ IDEA
# @Email ： 909709223@qq.com

'''
特征提取
'''

import math
from collections import Counter,defaultdict
# TLD
import tldextract
import numpy as np
from itertools import groupby

hmm_prob_threshold = -120
import pickle
import gib_detect_train

model_data = pickle.load(open('gib_model.pki', 'rb'))

def ave(array_):#sanity check for NaN
    if len(array_)>0:
        return array_.mean()
    else:
        return 0

def count_vowels(word):

    '''
    多少个元音a,e,i,o,u
    :param word:
    :return:
    '''

    vowels=list('aeiou')
    return sum(vowels.count(i) for i in word.lower())

def count_digits(word):

    '''
    多少个数字
    :param word:
    :return:
    '''

    digits=list('0123456789')
    return sum(digits.count(i) for i in word.lower())

def count_repeat_letter(word):

    '''
    多少个重复字符串
    :param word:
    :return:
    '''

    count = Counter(i for i in word.lower() if i.isalpha()).most_common()
    cnt = 0
    for letter,ct in count:
        if ct>1:
            cnt+=1
    return cnt


def consecutive_digits(word):

    '''
    多少个连续的数字
    :param word:
    :return:
    '''

    digit_map = [int(i.isdigit()) for i in word]
    consecutive=[(k,len(list(g))) for k, g in groupby(digit_map)]
    count_consecutive = sum(j for i,j in consecutive if j>1 and i==1)
    return count_consecutive

def consecutive_consonant(word):

    '''
    多少个连续的辅音
    :param word:
    :return:
    '''

    consonant = set(['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w','x', 'y', 'z'])
    digit_map = [int(i in consonant) for i in word]
    consecutive=[(k,len(list(g))) for k, g in groupby(digit_map)]
    count_consecutive = sum(j for i,j in consecutive if j>1 and i==1)
    return count_consecutive

def std(array_):

    '''
    检查NaN sanity check for NaN
    :param array_:
    :return:
    '''

    if len(array_)>0:
        return array_.std()
    else:
        return 0

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

def hmm_prob(domain):
    bigram = [''.join((i,j)) for i,j in bigrams(domain) if not i==None]
    prob = transitions[''][bigram[0]]
    for x in range(len(bigram)-1):
        next_step = transitions[bigram[x]][bigram[x+1]]
        prob*=next_step

    return prob

private_tld_file = open('private_tld.txt','r')
# black list for private tld
private_tld = set(f.strip() for f in private_tld_file)
private_tld_file.close()

n_gram_file = open('n_gram_rank_freq.txt','r')
gram_rank_dict = dict()
for i in n_gram_file:
    cat,gram,freq,rank = i.strip().split(',')
    gram_rank_dict[gram]=int(rank)
n_gram_file.close()

fi = open('training_w_tld.txt','r')
fw = open('features.txt','w')

feat_dict = dict()
'''
 feature extraction
 - bigrams of main domain
 - tld
 - length of main domain
 - entropy of main domain (unigram)

'''
header = 'ip\ttld\tentropy\tlen\tnorm_entropy\tvowel_ratio\tdigit_ratio\trepeat_letter\tconsec_digit\tconsec_consonant\tgib_value\thmm_log\tuni_rank\tbi_rank\ttri_rank\tuni_std\tbi_std\ttri_std\tprivate_tld\n'
fw.write('%s'%(header))

#pronounce detection
#google true
#baidu true
#123 true
#aaaf false
model_mat = model_data['mat']
threshold = model_data['thresh']

#load trans matrix for bigram markov model
transitions = defaultdict(lambda: defaultdict(float))

f_trans = open('trans_matrix.csv','r')
for f in f_trans:
    # key1 can be '' so rstrip() only
    key1,key2,value =f.rstrip().split('\t')
    value = float(value)
    transitions[key1][key2]=value

f_trans.close()

for f in fi:
    domain, tld = f.strip().split('\t')
    strip_domain = domain
    # user tld extractor for more precision
    ext = tldextract.extract(strip_domain)
    # remove non-ascii domain
    if len(ext.domain)>4 and ext.domain[:4]=='xn--':
        continue
    # add begin and end
    main_domain = '$'+ext.domain+'$'
    # ^ and $ of full domain name for HMM
    hmm_main_domain ='^'+domain.strip('.')+'$'
    tld = ext.suffix
    has_private_tld = 0
    # check if it is a private tld
    if tld in private_tld:
        has_private_tld = 1
        # quick hack: if private tld, use its last part of top TLD
        tld_list = tld.split('.')
        tld = tld_list[-1]
        # and overwrite the main domain
        main_domain = '$'+tld_list[-2]+'$'
    # extract the bigram
    bigram = [''.join(i) for i in bigrams(main_domain)]
    # extract the bigram
    trigram = [''.join(i) for i in trigrams(main_domain)]
    f_len = float(len(main_domain))
    # unigram frequency
    count = Counter(i for i in main_domain).most_common()
    # shannon entropy
    entropy = -sum(j/f_len*(math.log(j/f_len)) for i,j in count)
    unigram_rank = np.array([gram_rank_dict[i] if i in gram_rank_dict else 0 for i in main_domain[1:-1]])
    # extract the bigram
    bigram_rank = np.array([gram_rank_dict[''.join(i)] if ''.join(i) in gram_rank_dict else 0 for i in bigrams(main_domain)])
    # extract the bigram
    trigram_rank = np.array([gram_rank_dict[''.join(i)] if ''.join(i) in gram_rank_dict else 0 for i in trigrams(main_domain)])

    # linguistic feature: % of vowels, % of digits, % of repeated letter, % consecutive digits and % non-'aeiou'
    vowel_ratio = count_vowels(main_domain)/f_len
    digit_ratio = count_digits(main_domain)/f_len
    repeat_letter = count_repeat_letter(main_domain)/f_len
    consec_digit = consecutive_digits(main_domain)/f_len
    consec_consonant = consecutive_consonant(main_domain)/f_len

    # probability of staying in the markov transition matrix (trained by Alexa)
    hmm_prob_ = hmm_prob(hmm_main_domain)
    # probability is too low to be non-DGA
    if hmm_prob_<math.e**hmm_prob_threshold:
        hmm_log_prob = -999.
    else:
        hmm_log_prob = math.log(hmm_prob_)

    # advanced linguistic feature: pronouncable domain
    gib_value = int(gib_detect_train.avg_transition_prob(main_domain.strip('$'), model_mat) > threshold)
    try:
        fw.write('%s\t%s\t%.3f\t%.1f\t%.3f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%d\n'
                 %(domain,tld,entropy,f_len,entropy/f_len,vowel_ratio,
                   digit_ratio,repeat_letter,consec_digit,consec_consonant,gib_value,hmm_log_prob,
                   ave(unigram_rank),ave(bigram_rank),ave(trigram_rank),
                   std(unigram_rank),std(bigram_rank),std(trigram_rank),
                   has_private_tld)
                 )
    except UnicodeEncodeError:
        continue

fw.close()
fi.close()