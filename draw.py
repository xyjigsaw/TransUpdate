# Name: draw
# Author: Reacubeth
# Time: 2020/4/5 11:30
# Mail: noverfitting@gmail.com
# Site: www.omegaxyz.com
# *_*coding:utf-8 *_*

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import font_manager as fm, rcParams

cmp_name = 'transh'
path = 'result0409fromServer'
"""
      'new_triple': n_new_triple, 'used_eval_triple': n_used_eval_triple,
      'H_MR': head_meanrank_raw, 'H_h10': head_hits10_raw,
      'T_MR': tail_meanrank_raw, 'T_h10': tail_hits10_raw,
      'AVE_MR': (head_meanrank_raw + tail_meanrank_raw) / 2,
      'AVE_h10': (head_hits10_raw + tail_hits10_raw) / 2
"""

prop = fm.FontProperties(fname='times.ttf')

new_triple = []
used_eval_triple = []
head_meanrank = []
head_hits10 = []
tail_meanrank = []
tail_hits10 = []
ave_meanrank = []
ave_hits10 = []


with open(path + '/' + cmp_name + '/' + "my_raw.txt", "r") as f:
    for line in f.readlines():
        nt, uet, hm, hh, tm, th, am, ah = line.strip('\n').split(',')
        new_triple.append(float(nt))
        used_eval_triple.append(float(uet))
        head_meanrank.append(float(hm))
        head_hits10.append(float(hh))
        tail_meanrank.append(float(tm))
        tail_hits10.append(float(th))
        ave_meanrank.append(float(am))
        ave_hits10.append(float(ah))


new_triple_t = []
used_eval_triple_t = []
head_meanrank_t = []
head_hits10_t = []
tail_meanrank_t = []
tail_hits10_t = []
ave_meanrank_t = []
ave_hits10_t = []


with open(path + '/' + cmp_name + '/' + cmp_name + '_raw.txt', "r") as f:
    for line in f.readlines():
        nt, uet, hm, hh, tm, th, am, ah = line.strip('\n').split(',')
        new_triple_t.append(float(nt))
        used_eval_triple_t.append(float(uet))
        head_meanrank_t.append(float(hm))
        head_hits10_t.append(float(hh))
        tail_meanrank_t.append(float(tm))
        tail_hits10_t.append(float(th))
        ave_meanrank_t.append(float(am))
        ave_hits10_t.append(float(ah))

'''
# 使用两次 bar 函数画出两组条形图
plt.bar(new_triple_t, height=ave_meanrank_t, width=2, color='b', label='transE')
plt.bar(new_triple, height=ave_meanrank, width=2, color='g', label='my')

plt.legend()  # 显示图例
plt.xticks(new_triple, new_triple)
plt.ylabel('MeanRank')  # 纵坐标轴标题
plt.title('New triples')  # 图形标题

plt.show()
'''

fig, ax1 = plt.subplots()
ax1.plot(new_triple, ave_meanrank, label='My', marker='+')
ax1.plot(new_triple, ave_meanrank_t, label='transE', marker='*')
ax1.set_xlabel('New Triples')
ax1.set_ylabel('MeanRank')
#ax2 = ax1.twinx()
#ax2.plot(new_triple, ave_hits10, label='My', marker='+')
#ax2.plot(new_triple, ave_hits10_t, label='transE', marker='*')
#ax2.set_ylabel('Hit@10')

plt.legend()
plt.show()
