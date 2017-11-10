'''
本文件主要用于生成公司的预分词文件，便于之后处理
'''

from util import Company_Cut, clean_punc, find_range
import jieba
from time import time
import multiprocessing

org_cut = Company_Cut()


class Company_PreProcessing(object):  # 创建一个迭代器，送入公司
    def __init__(self, dirname, cal_range=None):
        self.dirname = dirname
        self.range = cal_range
        self.count = 0

    def __iter__(self):
        for line in open(self.dirname):
            if self.range:
                if self.range[0] <= self.count <= self.range[1]:
                    listline = line.strip()
                    yield listline
                elif self.count > self.range[1]:
                    break
            else:
                listline = line.strip()
                yield listline
            self.count += 1


def company_pre_seg(index_range):
    '''
    对公司进行预先的分词
    '''
    (start_index, end_index) = index_range
    cal_num = end_index - start_index
    count = 1
    segged_company_text_list = []
    start = time()
    company = Company_PreProcessing('/Users/pp/pycharmprojects/Data/Companys/company.txt', index_range)
    for line in company:
        if not line:
            continue
        cut_list, split_result = org_cut.absolutely_cut(clean_punc(line))
        segged_company_text = ' '.join(cut_list)
        segged_company_text_list.append(segged_company_text.strip())
        if count % 10000 == 0:
            print('processed', count, count / cal_num, 'cost', time() - start, 'range', index_range)
        count += 1
    print('task finished')
    return segged_company_text_list


if __name__ == '__main__':
    company_len = Company_PreProcessing('/Users/pp/pycharmprojects/Data/Companys/company.txt')
    for i in company_len:
        pass
    company_num = company_len.count
    # company_num=10000
    print(org_cut.cut_adjust(list(jieba.cut('中国移动通信集团'))))
    cpu_count = 6
    index_range = find_range(cpu_count, company_num)
    print(index_range)
    pool = multiprocessing.Pool(processes=cpu_count)
    result = pool.map(company_pre_seg, index_range)

    # 写入文件
    with open('/Users/pp/pycharmprojects/Data/Companys/Company_PreSeg.txt', 'w') as f:
        for patch in result:
            for line in patch:
                f.write(line + '\n')
