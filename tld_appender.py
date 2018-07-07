#!/usr/bin/python
# -*- coding:utf8 -*-
# @Author : tuolifeng
# @Time : 2018/6/30 上午10:33
# @File : tld_appender.py
# @Software : IntelliJ IDEA
# @Email ： 909709223@qq.com
# 解析文本中的顶级域名ccTLD

def max_match(domain, tlds):
    match = [i for i in tlds if i in domain]
    if len(match)>0:
        for i in sorted(match,key=lambda x: len(x), reverse = True):
            # longest and matches the end of the domain
            if i == domain[-(len(i)):]:
                return i
    else: return 'NONE'

tld_file = open('tld_list.txt','r')
# for domain match, add dot as prefix and postfix
tld_list = list('.'+t.strip().strip('.')+'.' for t in tld_file)
tld_file.close()

# fi = open('1393459321.nps_malware_dga_training_set.txt','r')
# fi = open('expanded_training_set.txt','r')
fi = open('domain_list.txt','r')

fw = open('training_w_tld.txt','w')

for f in fi:
    domain = f.strip()
    match = max_match(domain, tld_list)
    fw.write('%s\t%s\n'%(domain,match))

fw.close()
fi.close()