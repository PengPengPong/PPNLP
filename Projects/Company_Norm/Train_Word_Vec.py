'''
本文件用于根据已有的预处理的公司分词文件，训练词向量
'''

import logging
import gensim
from time import time
import multiprocessing


class MySentences_PreProcessing(object):  # 创建一个迭代器，往模型中送入处理后的句子文本
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for line in open(self.dirname):
            listline = line.strip().split()
            yield listline


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


inpath = '/Users/pp/pycharmprojects/Data/Companys/Company_PreSeg.txt'
model_path = '/Users/pp/pycharmprojects/data/归一化标注//Users/pp/pycharmprojects/nlp/projects/'
model_name = 'model_company(1000,40,5,10)_20171109_absolute_cut'

start = time()
sentences = MySentences_PreProcessing(inpath)  # 输入是句子的序列. 每个句子是一个单词列表
model = gensim.models.Word2Vec(sentences,
                               size=1000,  # 词向量的维度
                               window=40,  # 上下文字数窗口
                               min_count=5,  # 最小处理频数，词频低于此的词语将直接跳过不处理
                               iter=10,  # 迭代次数，默认为5
                               workers=multiprocessing.cpu_count())  # 并行用的处理器数

model.save(model_path+model_name)  # 储存模型

end = time()
print("Total procesing time: %d seconds" % (end - start))
