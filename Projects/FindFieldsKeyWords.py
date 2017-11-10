# 导入数据

print('start to load data')

import re
import jieba
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from MyNLP import *
import bottleneck
import thulac  # 另外一套分词组件，目的是不引入自定义词表


functions = {}
with open('/Users/pp/pycharmprojects/nlp/job function.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        job, label = line.strip().lower().split(',')
        if label == 'job':
            new_func = job
            functions.setdefault(new_func, set())
        else:
            functions[new_func].add(job)

model1 = Word2Vec.load(
    '/Users/pp/pycharmprojects/nlp/Word2Vec/model_jobtitle_Alpha_IFC_Segment_CleanWord_Job_User_Dict(1000,40,5,10)')
# model1 = Word2Vec.load('model_jobtitle_Alpha_IFC_Segment_CleanWord_WithProject(1000,40,5,5)')
jieba.load_userdict('/Users/pp/pycharmprojects/nlp/Job_User_Dict.txt')
thuseg = thulac.thulac(seg_only=True)  # 默认模式

with open('/Users/pp/pycharmprojects/nlp/job_stop_word.txt') as f:
    job_stop_words = f.read().split('\n')
with open('/Users/pp/pycharmprojects/nlp/dont stop me.txt') as f:
    job_dont_stop_words = set(f.read().split('\n'))
with open('/Users/pp/pycharmprojects/nlp/locations.txt') as f:
    locations = set(f.read().split('\n'))

vac_idf = {}
for line in open("/Users/pp/pycharmprojects/nlp/Jobtitle_idf_Alpha_IFC_Clean.txt"):
    # for line in open("/Users/pp/pycharmprojects/nlp/idf_Full_WithFuncDict.txt"):
    sline = line.split()
    vac_idf.setdefault(sline[0], float(sline[1]))  # 导入预先计算好的IDF

vac_pos = {}
# with open('/Users/pp/pycharmprojects/nlp/job_pos_NotCleared.txt','r') as f:
with open('/Users/pp/pycharmprojects/nlp/job_pos.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        job, pos = line.strip().lower().split()
        vac_pos.setdefault(job, float(pos))

vac_info = {}
with open('/Users/pp/pycharmprojects/nlp/Word_Semantic_Richness.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        word, freq, entropy, coagulation = line.strip().lower().split()
        vac_info.setdefault(word, {'ent': float(entropy), 'freq': int(freq)})


totalcount=2676891

import json
from time import time


# 目前的JD的英文部分的英文都是粘结在一起的，之后等源数据更新后再处理

@profile
def KeyWordIDF():
    KeywordInVec = {}
    count = 0

    start = time()

    for jd_str in open("/Users/pp/pycharmprojects/Data/JDs/jobui_total.json"):

        count += 1
        if count % 10000 == 0:
            print('processed', count / totalcount, 'cost', time() - start)
            break

        #         if count>12991:
        #             break

        #         if count!=12991:
        #             continue


        jd_des = ''
        jd = json.loads(jd_str)
        if not jd:
            continue
        jd_responsibility = jd.get('job_responsibility', '')
        jd_requirement = jd.get('job_requirement', '')
        jd_title = jd.get('title', '')
        if jd_responsibility:
            jd_des += jd_responsibility
        if jd_requirement:
            jd_des += jd_requirement

        # if 'logistics' in jd_des:
        #             print(jd_str)
        #             print(count)
        #             print(jd_title)
        #             print(jd_des)
        #         else:
        #             continue





        vac_list = []
        for similist in Find_KeyWords(jd_title, vac_info, vac_pos, job_stop_words, job_dont_stop_words, locations):
            for jobseg, weight in similist:
                try:
                    vacarray = np.array(model1[jobseg])
                    vac_list.append(vacarray / np.linalg.norm(vacarray) * weight)
                except:
                    continue

        if not vac_list:
            continue
        jobvac = np.array(vac_list).sum(0)  # 职位向量和
        jobvac = jobvac / np.linalg.norm(jobvac)

        #         if np.isnan(jobvac).any():
        #             print('jobvac',jobvac)
        #             print(count)
        #             print(jd_title)
        #             return None

        for des_seg in set(jieba.cut(jd_des)):
            if len(des_seg) > 1:
                des_seg = des_seg.lower()
                if des_seg not in KeywordInVec:
                    KeywordInVec.setdefault(des_seg, {'vac': np.zeros_like(jobvac), 'count': 0})
                KeywordInVec[des_seg]['vac'] += jobvac
                KeywordInVec[des_seg]['count'] += 1
        # break

    Keyword_idf = {}
    for word, info in KeywordInVec.items():
        Keyword_idf.setdefault(word, np.linalg.norm(info['vac']) / info['count'])

    # print(Keyword_idf)
    return Keyword_idf

KeyWordIDF()