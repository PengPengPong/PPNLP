# 导入数据

print('start to load data')

import re
import jieba
import pandas as pd
import numpy
from gensim.models import Word2Vec
from MyNLP import *
import bottleneck
import thulac  # 另外一套分词组件，目的是不引入自定义词表

# 导入词向量模型
model = Word2Vec.load(
    '/Users/pp/pycharmprojects/nlp/Word2Vec/model_jobtitle_Alpha_IFC_Segment_CleanWord_Job_User_Dict(1000,40,5,10)')
# model = Word2Vec.load('/Users/pp/pycharmprojects/nlp/Word2Vec/WordVec_CV&JD(200,10,5,5)')


# 导入自定义词库
jieba.load_userdict('/Users/pp/pycharmprojects/nlp/Job_User_Dict.txt')

# 导入停词
with open('/Users/pp/pycharmprojects/nlp/job_stop_word.txt') as f:
    job_stop_words = f.read().split('\n')

# 导入误删除停词
with open('/Users/pp/pycharmprojects/nlp/dont stop me.txt') as f:
    job_dont_stop_words = set(f.read().split('\n'))

# 导入地区
with open('/Users/pp/pycharmprojects/nlp/locations.txt') as f:
    locations = set(f.read().split('\n'))

# 导入预先算好的词的位置信息
vac_pos = {}
with open('/Users/pp/pycharmprojects/nlp/job_pos.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        job, pos = line.strip().lower().split()
        vac_pos.setdefault(job, float(pos))

# 导入预先算好的词的语义丰富程度信息
vac_info = {}
with open('/Users/pp/pycharmprojects/nlp/Word_Semantic_Richness.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        word, freq, entropy, coagulation = line.strip().lower().split()
        vac_info.setdefault(word, {'ent': float(entropy), 'freq': int(freq)})

# 导入预先算好的词的idf信息
Keyword_idf = {}
for lines in open('/Users/pp/pycharmprojects/Data/JDs/JD_promoted_IDF.txt'):
    try:
        [word, idf] = lines.strip().split()
    except:
        continue
    idf = float(idf)
    Keyword_idf.setdefault(word, idf)


def Get_JobVac(JobTitle):
    '''
    Intro:
        · 获取职位向量
    Args:
        · JobTitle:职位
    Return:
        · 职位对应的向量
    '''
    vac_list = []
    job_vac = []
    for keyword in Find_KeyWords(JobTitle, vac_info, vac_pos, job_stop_words, job_dont_stop_words, locations):
        for jobseg, weight in keyword:

            if jobseg in ['管理', '工程', '开发']:
                continue

            try:
                vacarray = model[jobseg]  # BUG:1,已修改
                vac_list.append(vacarray / np.linalg.norm(vacarray) * weight)
            except:
                for subseg in thuseg.cut(jobseg, text=True).split():

                    if subseg in ['管理', '工程', '开发']:
                        continue

                    try:
                        vacarray = np.array(model[subseg])
                        vac_list.append(vacarray / np.linalg.norm(vacarray) * weight)
                    except:
                        continue

        if vac_list:
            sum_vac = np.array(vac_list).sum(0)
            job_vac.append(sum_vac / np.linalg.norm(sum_vac))  # 预先归一化，降低求向量夹角的时间复杂度
    return job_vac


def Job_Similarity(JobTitle1, JobTitle2):
    '''
    Intro:
        · 计算职位之间的相似度
    Args:
        · JobTitle1:职位文本1
        · JobTitle2:职位文本2
    Return:
        · 职位的相似度；如果判定某个词不是一个职位词，则返回0

    '''
    simi_list = [0]
    for jobvac1 in Get_JobVac(JobTitle1):
        for jobvac2 in Get_JobVac(JobTitle2):
            try:
                simi_list.append(CosSimilarity(jobvac1, jobvac2))
            except:
                continue
    return max(simi_list)


def Get_JD_KeyWords(Text, Percentage=0.2):
    keyword_weight = {}
    for i in jieba.cut(Text.strip().lower()):
        if len(i) > 1 and not i.encode('utf8').isdigit() and i not in ['管理', '工程', '开发']:
            keyword_weight[i] = keyword_weight.get(i, 0) + 1

    # 计算权重
    for keyword, freq in keyword_weight.items():
        keyword_weight[keyword] = freq * Keyword_idf.get(keyword, 1) ** 2

    keyword_weight_sorted = sorted(keyword_weight.items(), key=lambda x: x[1], reverse=True)
    return keyword_weight_sorted[:math.ceil(len(keyword_weight_sorted) * Percentage)]


def Get_DutyVac(Text, Percentage=0.2):
    # 获取一段文本的“关键词-向量”对

    duty_vac = []
    for keyword, weight in Get_JD_KeyWords(Text, Percentage):
        try:
            duty_vac.append((keyword, model[keyword] / np.linalg.norm(model[keyword])))  # 预先归一化，降低求向量夹角的时间复杂度
        except:
            continue
    return duty_vac

# 导入职能表及其对应的词向量
# TODO:切换到数据库时，这里需要重新写
function_vec={}
with open('/Users/pp/pycharmprojects/nlp/job function.txt','r') as f:
    lines=f.readlines()
    for line in lines:
        job,label=line.strip().lower().split(',')
        if label=='job':
            new_func=job
            function_vec.setdefault(new_func,[])
        else:
            subfunc_vec=Get_JobVac(job)
            if subfunc_vec:
                sum_vac=np.array(subfunc_vec).sum(0)
                function_vec[new_func].append(sum_vac/np.linalg.norm(sum_vac))


# 职位与订阅职能的匹配度

@profile
def Job_Func_Similarity(JobTitle, JobDuty, Subscription, Percentage=0.2):
    '''
    Intro:
        · 计算一个职位和用户订阅的职能匹配度
    Args:
        · JobTitle:职位
        · JobDuty:职位描述
        · Subscription:用户订阅的职能列表
    Return:
        · 用户订阅的职能中与该职位最大的相似度
    '''

    # 获取职位向量
    jobvac = Get_JobVac(JobTitle)
    # 获取职位描述向量集合
    dutyvac = Get_DutyVac(JobDuty, Percentage)

    # 如果职位向量为空
    if not jobvac:

        for i in ['助理', '文员', '助手']:
            if i in JobTitle:
                jobvac = Get_JobVac('文秘')
                break

    max_job_simi = [0]
    if not jobvac:
        JobTitle = JobTitle.lower()
        for subscription_job in Subscription:
            for job in functions[subscription_job]:
                job = job.lower()
                if job in JobTitle:
                    max_job_simi = [1]

    # 计算职位名相似度
    for jobvac_seg in jobvac:
        for function in Subscription:
            job_simi_subfunc = [0]
            for func_vec in function_vec[function]:
                try:
                    job_simi_subfunc.append(CosSimilarity(jobvac_seg, func_vec, PreNormalized=True))
                except:
                    continue
            max_job_simi.append(max(job_simi_subfunc))  # 职位名与某职能的相似度

    # 计算每个关键词与所有订阅职能中最大的相似度
    max_duty_simi = [0]
    for keyword, dutyvac_seg in dutyvac:
        duty_simi_subfunc = [0]
        for function in Subscription:
            for func_vec in function_vec[function]:
                try:
                    duty_simi_subfunc.append(
                        CosSimilarity(dutyvac_seg, func_vec, PreNormalized=True))  # 某关键词与所有订阅子职能的相关度
                except:
                    continue
                    #         print(duty_simi_subfunc)
        max_duty_simi.append(max(duty_simi_subfunc))

    # print('max_duty_simi',max_duty_simi)

    # 对关键词的结果进行调整：由于取的是最大的，可能会由于一些偶尔的噪音，导致结果偏大。尝试去除与整体情况格格不入的最大值。
    max_duty_simi_sorted = sorted(max_duty_simi, reverse=True)
    max_num = 2 if len(max_duty_simi_sorted) > 10 else 1
    for i in range(max_num):  # 如果关键词数量大于10，则排除两次，否则一次
        try:
            # 最大值与第N大值相差超过两倍，且最大值不出现在权重最高的关键词，且关键词数量大于5，则执行最大值排除
            if max_duty_simi_sorted[0] > 2 * max_duty_simi_sorted[max_num] and max_duty_simi_sorted[0] != max_duty_simi[
                0] and len(max_duty_simi_sorted) > 5:
                del max_duty_simi_sorted[0]
                del max_duty_simi[0]
        except:
            continue
    return max((max(max_job_simi), max(max_duty_simi)))

title='城市公司总经理'
duty='''
"主要全面负责城市公司各门店的运营管理
（1）组织新门店的筹备及老门店运营管理（包括物料、人员、流程、制度、经济指标的计划分解、客户关系经营、社群活动企划执行等项工作）；对城市公司各门店的招租指标达成、社群文化建设、团队可持续发展负责；
（2）主持城市公司开发团队完成新项目勘察，项目谈判、项目签约及业主关系维护。按目标和流程报总公司会审，办理后续签约、交接等事项；
（3）组织目前期勘察及项目整体方案的讨论、设计及决策；全面监管各新项目的工程进度、质量安全，对项目按时保质交付负责；维护业主及周边邻里关系；
（4）根据总公司品牌规划，负责城市公司市场品牌传播策略的组织执行，在品牌中心的协助下进行品牌宣传，协调商业资源合作、媒体公关维护等工作；
（5）主持推动城市公司关键管理流程和规章制度的建设，及时进行组织和流程的优化调整、领导营造企业文化氛围、完善企业识别系统、塑造和强化公司价值观；"
'''

Job_Func_Similarity(title,duty,['产品','研发','市场','运营'])