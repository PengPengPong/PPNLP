'''
本文件主要用于给人工词表打上标记，检验人工词表的正确性
也用于给模型中的所有词汇预先打一个标记
'''
from util import Company_Cut, find_range
from time import time
import gensim
from gensim import matutils
import numpy as np
import multiprocessing
import jieba

org_cut = Company_Cut()
model_path = '/Users/pp/pycharmprojects/nlp/projects/'
model_name = 'model_company(1000,40,5,10)_20171109_absolute_cut'
model = gensim.models.Word2Vec.load(model_path + model_name)
vocab = list(model.wv.vocab.keys())


def cal_cluster_vec(cluster_set):
    '''
    生成一类词表的词向量
    :param cluster_set: 词表
    :return: 对应的词向量
    '''
    cluster_vec = []
    for clus in cluster_set:
        try:
            cluster_vec.append(model[clus])
        except:
            continue
    cluster_vec = np.array(cluster_vec)
    return cluster_vec


ind_vec = cal_cluster_vec(org_cut.ind_set)
name_vec = cal_cluster_vec(org_cut.name_set)
loc_vec = cal_cluster_vec(org_cut.loc_set)


def cluster_top_simi(cluss_vec, single_vec, topn=5):
    '''
    计算一个词处于某一类的概率
    '''
    simi_vec = np.dot(cluss_vec, single_vec)
    topn_simi = simi_vec[matutils.argsort(simi_vec, topn=topn, reverse=True)]
    if topn_simi[0] > 0.99:  # 如果存在完全匹配的，则直接返回完全匹配的
        return 1
    else:
        return topn_simi.mean(0)


def find_company_seg_type(company_seg):
    '''
    利用词向量获取seg的类型
    TODO:以后可逐步优化，利用机器分类模型直接分类
    '''
    simi_dict = ['name', 'loc', 'ind']
    simi_result = []
    try:
        seg_vec = model[company_seg]
    except:
        return 'none'  # 词表中没出现则标记为none
    simi_result.append(cluster_top_simi(name_vec, seg_vec))
    simi_result.append(cluster_top_simi(loc_vec, seg_vec))
    simi_result.append(cluster_top_simi(ind_vec, seg_vec))
    simi_result = np.array(simi_result)
    # 如果存在多个1，则字号变为0
    if simi_result[0] == simi_result[1] == 1 or simi_result[0] == simi_result[2] == 1:
        simi_result[0] = 0

    return simi_dict[simi_result.argmax()]


def machine_tag(result_file, to_tag_list):
    '''
    利用机器给人工词表中的词打上标签
    :param result_file:
    :param to_tag_list:
    :return:
    '''
    count = 0
    start = time()
    with open(result_file, 'w') as f:
        for to_tag in to_tag_list:
            tag = find_company_seg_type(to_tag)
            f.write(to_tag + '\t' + tag + '\n')
            count += 1
            if count % 1000 == 0:
                print('processed', count, 'cost', time() - start)


def vocab_pre_tag(index_range):
    vacab_tag = {}
    range_len = index_range[1] - index_range[0]
    count = 0
    start = time()
    for voc in vocab[index_range[0]:index_range[1]]:
        vacab_tag[voc] = find_company_seg_type(voc)
        count += 1
        if count % 10000 == 0:
            print('processed' / range_len, count, 'cost', time() - start)
    return vacab_tag


def generate_model_vac_tag():
    # 对模型中的每个词语预先算好属性，节约时间
    # 利用并行进行计算
    cpu_count = 4
    vocab_num = len(vocab)
    # vocab_num=1000
    index_range = find_range(cpu_count, vocab_num)
    print(index_range)
    pool = multiprocessing.Pool(processes=cpu_count)
    result = pool.map(vocab_pre_tag, index_range)

    # 存入文件，方便以后调用
    with open('/Users/pp/pycharmprojects/data/归一化标注//Users/pp/pycharmprojects/data/归一化标注/' + model_name + '.txt',
              'w') as f:
        for res in result:
            for k, v in res.items():
                f.write(k + '\t' + v + '\n')


def short_cut_for_ind():
    # 对行业词表进行短分词
    short_seg = []
    for line in open('/Users/pp/pycharmprojects/data/归一化标注/行业词表.txt', 'r'):
        line = line.strip()
        segs = org_cut.cut_adjust(list(jieba.cut(line)))
        short_seg.extend(segs)
    short_seg = set(short_seg)
    with open('/Users/pp/pycharmprojects/data/归一化标注/行业词表_短分词.txt', 'w') as f:
        for i in short_seg:
            f.write(i + '\n')


if __name__ == '__main__':
    # short_cut_for_ind()

    # 标注字号表
    to_tag_list = [line.strip() for line in open('/Users/pp/pycharmprojects/data/归一化标注/字号词表.txt', 'r')]
    machine_tag('/Users/pp/pycharmprojects/data/归一化标注/字号词表_机器标注_原始第二次.txt', to_tag_list)

    # 标注行业表
    to_tag_list = [line.strip() for line in open('/Users/pp/pycharmprojects/data/归一化标注/行业词表_短分词.txt', 'r')]
    machine_tag('/Users/pp/pycharmprojects/data/归一化标注/行业词表_机器标注_原始第二次.txt', to_tag_list)
