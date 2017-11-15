import math
import jieba
import re
import numpy as np
import gensim
from gensim import matutils


def clean_punc(String):  # 删除所有非空格的标点符号以及数字，含中英文
    s = String
    text = re.sub(u"[\s\.\!\/_,&$~%^*()+\"\'\\\|\-<;>\[\]|:￥《》～“”【】￡︳∕／＼∕＿：――·・﹑﹕；﹡＊＆｜︱￥#……（）［］．！｛｝＜＞﹣——－—＝＋﹐，。？、]",
                  "", s)
    return text


def find_range(peices, total):
    '''
    mapping时候用于生成作用范围
    :param peices:
    :param total:
    :return:
    '''
    params_range = []
    begin = 1
    step = math.ceil(total / peices)
    while begin < total:
        params_range.append((begin, min(begin + step, total)))
        begin += step + 1
    return params_range


class Company_Cut():
    def __init__(self):
        self.model = self.load_word_vec_model()
        self.one_word_pos = self.load_one_word_pos()
        self.vac_tag = self.load_vac_tag()
        self.loc_set = self.load_loc_set()
        self.loc_suffix_set = self.load_loc_suffix_set()
        self.ind_set = self.load_ind_set()
        self.name_set = self.load_name_set()
        self.ind_vec = self.cal_cluster_vec(self.ind_set)
        self.name_vec = self.cal_cluster_vec(self.name_set)
        self.loc_vec = self.cal_cluster_vec(self.loc_set)
        self.company_suffix = self.load_company_suffix()
        self.adjust_jieba_dict()

    def load_word_vec_model(self):
        model_path = '/Users/pp/pycharmprojects/nlp/projects/'
        model_name = 'model_company(1000,40,5,10)_20171111_absolute_cut'
        model = gensim.models.Word2Vec.load(model_path + model_name)
        model.wv.init_sims(True)  # 预先将模型中的向量归一化
        return model

    def load_one_word_pos(self):
        # 导入人工标注的词的合并信息
        one_word_pos = {}
        for line in open('/Users/pp/pycharmprojects/data/归一化标注/one_len_word_freq.txt'):
            [word, freq, pos] = line.split()
            one_word_pos[word] = int(pos)
        return one_word_pos

    def load_vac_tag(self):
        # 导入机器打的整个词向量词汇表中的词语属性标签
        vac_tag = {}
        for line in open(
                '/Users/pp/pycharmprojects/data/归一化标注/model_company(1000,40,5,10)_20171111_absolute_cut_模型下所有词语类型机器标注_无强制匹配.txt',
                'r'):
            [vac, tag] = line.strip().split()
            vac_tag[vac] = tag
        return vac_tag

    def load_loc_set(self):
        # 导入地区
        with open('/Users/pp/pycharmprojects/nlp/locations.txt') as f:
            loc_set = set(f.read().split('\n'))
        return loc_set

    def load_loc_suffix_set(self):
        # 导入地区后缀
        with open('/Users/pp/pycharmprojects/Data/归一化标注/loc_suffix_strict.txt') as f:
            loc_suffix_set = set(f.read().split('\n'))
        return loc_suffix_set

    def load_ind_set(self):
        # 导入行业词
        with open('/Users/pp/pycharmprojects/data/归一化标注/行业词表_最终确认.txt') as f:
            ind_set = set(f.read().split('\n'))
        return ind_set

    def load_name_set(self):
        # 导入字号词
        with open('/Users/pp/pycharmprojects/data/归一化标注/字号词表_最终确认.txt') as f:
            name_set = set(f.read().split('\n'))
        return name_set

    def load_company_suffix(self):
        # 导入公司suffix词典
        from collections import OrderedDict
        company_suffix = OrderedDict()
        for line in open('/Users/pp/pycharmprojects/data/归一化标注/company_suffix.txt', 'r'):
            suffix, tag = line.strip().split()
            company_suffix[suffix] = tag
        return company_suffix

    # 预先计算行业词、字号词、地区词的词向量，储存
    def cal_cluster_vec(self, cluster_set):
        cluster_vec = []
        for clus in cluster_set:
            try:
                cluster_vec.append(self.model[clus])
            except:
                continue
        cluster_vec = np.array(cluster_vec)
        return cluster_vec

    def adjust_jieba_dict(self):
        '''
        对结巴自定义的词库进行调整
        :return:
        '''
        # 初始化jieba的词典
        jieba.dt.check_initialized()
        # 先删除jieba原有词库中不该存在的词
        jieba_dict_to_be_del = []
        loc_suffix_to_be_del = set()
        for line in open('/Users/pp/pycharmprojects/Data/归一化标注/user_loc_suffix_to_be_del.txt', 'r'):
            # 从company_suffix拷贝而来，略作修改
            loc_suffix_to_be_del.add(line.strip())

        for word, freq in jieba.dt.FREQ.items():
            # 小于等于3字的不处理
            if len(word) <= 3:
                continue
            # 大于等于5个字的全部删除(容易导致地区词也被删除)
            elif len(word) >= 5:
                jieba_dict_to_be_del.append(word)
            # 4个字的，如果含公司后缀，则删除
            else:
                for suf in loc_suffix_to_be_del:
                    #                     if len(suf) == 1 and len(word) == 3:
                    #                         continue
                    if word[-len(suf):] == suf:
                        jieba_dict_to_be_del.append(word)
                        break
        for i in jieba_dict_to_be_del:
            jieba.dt.FREQ.pop(i)
        # 再导入用户自定义词表：不直接调用jieba.loads，因为如果用户词表中包含jieba中已有的词，词频会覆盖，这是不希望发生的
        for line in open('/Users/pp/pycharmprojects/Data/Companys/User_Dict_For_CompanyNorm.txt', 'r'):
            [word, freq] = line.strip().split()
            jieba.dt.FREQ.setdefault(word, int(freq))

    def clear_company_suffix(self, company):
        '''
        清除组织后缀词
        '''
        main_part = company
        for suf in self.company_suffix.keys():
            if company[-len(suf):] == suf:
                main_part = company[:-len(suf)]
                break
        return main_part

    def is_loc(self, word):
        if not word:
            return False
        elif self.vac_tag.get(word, "") == 'loc' or word in self.loc_set or (word[-1:] in self.loc_suffix_set) or (
                    word[-2:] in self.loc_suffix_set):
            return True
        else:
            return False

    def is_ind(self, word):
        if self.vac_tag.get(word, "") == 'ind':
            return True
        else:
            return False

    def is_vac(self, word):
        if word in jieba.dt.FREQ or word in self.vac_tag:
            return True
        else:
            return False

    def cut_adjust(self, seg_list):
        # TODO:看看能不能从决策树模型发展出一套吸引力/排斥力模型出来，感觉很好玩的样子。在归一化弄完之后尝试。
        adjusted_seg = []
        global skip
        skip = 0

        def forward_merge():
            adjusted_seg.append(adjusted_seg[-1] + seg_list[j])
            del adjusted_seg[-2]

        def backward_merge():
            global skip
            if j == len(seg_list) - 1:
                adjusted_seg.append(seg_list[j])  # 后面没有词了
            if j < len(seg_list) - 1:
                adjusted_seg.append(seg_list[j] + seg_list[j + 1])
                skip = 1
            if j < len(seg_list) - 2 and len(seg_list[j + 2]) == 1:
                del adjusted_seg[-1]
                adjusted_seg.append(seg_list[j] + seg_list[j + 1] + seg_list[j + 2])
                skip = 2

        # 合并策略
        for j in range(len(seg_list)):
            if skip:
                skip = max(0, skip - 1)
                continue
            if len(seg_list[j]) == 1:  # 单字

                # 如果后面也是单字，则持续合，直到下标越界或者不是单字为止
                i = 1  # 记录往后看的词数
                one_merge = seg_list[j]  # 将单字放入待合并列表
                while j + i < len(seg_list) and len(seg_list[j + i]) == 1:
                    one_merge += seg_list[j + i]
                    skip += 1
                    i += 1
                if len(one_merge) > 1:  # 如果确实合并了后面的单字
                    adjusted_seg.append(one_merge)
                    continue

                if j == 0:  # 如果是第一个字,直接与后面的合

                    backward_merge()

                elif j == 1 and self.one_word_pos.get(seg_list[j], 0) == 1:
                    # 第二个字，是人工标注的需要向后合并的，向后合并
                    backward_merge()
                elif j == 1 and self.one_word_pos.get(seg_list[j], 0) == -1:
                    # 第二个字，是人工标注的需要向前合并的，向前合并
                    forward_merge()
                elif (j == 1 and not self.is_loc(seg_list[j - 1])) or (
                                    j == 1 and len(seg_list) >= 3 and self.is_ind(seg_list[j + 1])) or (
                                    j == 2 and not self.is_loc(seg_list[j - 1]) and not self.is_loc(seg_list[j - 2])):
                    # 第二个字，前一个不是地区;或者第二个字，(前一个是两字地区,此条删除），后一个是行业词；或者第三个字，前两个都不是地区;与前面的合
                    forward_merge()

                elif j == len(seg_list) - 1 and seg_list[j] in self.company_suffix.keys():
                    # 如果是最后一个字，且是组织后缀（暂时不扩展到全位置后缀），不做合并
                    adjusted_seg.append(seg_list[j])

                elif j > 0:
                    if self.is_loc(seg_list[j - 1]):  # 如果前一个词是地区词，则与后一个合并
                        # 但是如果后一个是大于等于4个字的，则与前一个合并
                        if j < len(seg_list) - 1 and len(seg_list[j + 1]) >= 4:
                            forward_merge()

                        # 如果该词是人工标注需要向前合并的，且，(往前两个词也是地区词 或 后面一个词是行业词)，则地区可能判断出错，与前一个合并
                        elif self.one_word_pos.get(seg_list[j], 0) == -1 and (
                                    (j > 1 and self.is_loc(seg_list[j - 2])) or (
                                                j < len(seg_list) - 1 and self.is_ind(seg_list[j + 1]))):
                            forward_merge()

                        # 如果该词后面一个词是行业词或者为3个字（4个字的场景已经在前面判断了），且，往前两个也是地区词，则地区可能判断出错，与前一个合并
                        elif j > 1 and self.is_loc(seg_list[j - 2]) and j < len(seg_list) - 1 and (
                                    self.is_ind(seg_list[j + 1]) or len(seg_list[j + 1]) == 3):
                            forward_merge()

                        # 如果该词后面一个词也是地区词，则谁词频高谁更可能是地区.如果相等则往后合并
                        elif j < len(seg_list) - 1 and self.is_loc(seg_list[j + 1]):
                            freq_before = jieba.dt.FREQ.get(seg_list[j - 1], 0)
                            freq_after = jieba.dt.FREQ.get(seg_list[j + 1], 0)
                            if freq_before >= freq_after:
                                backward_merge()
                            else:
                                forward_merge()

                        # 后一个字数小于4
                        elif j < len(seg_list) - 1 and len(seg_list[j + 1]) < 4:
                            backward_merge()

                        else:
                            adjusted_seg.append(seg_list[j])

                    elif j > 1 and self.is_loc(seg_list[j - 2]):  # 如果往前数两个是地区词，则大概率与前一个合并
                        # 如果人工标注下也表明其应当向前合并，则直接向前合并：
                        if self.one_word_pos.get(seg_list[j], 0) == -1:  # 如果是人工标注的需要往前合并的位置
                            forward_merge()

                        # 如果和后面的词合并了是一个行业词，（或者是人工标注的需要往后合并：未添加），则与后面合并，否则和前面合并
                        elif j < len(seg_list) - 1 and self.is_ind(
                                        seg_list[j] + self.clear_company_suffix(seg_list[j + 1])):
                            backward_merge()

                        # 如果前面是行业词，且后面不是行业词，则向后合并：
                        elif j > 0 and self.is_ind(seg_list[j - 1]) and j < len(seg_list) - 1 and not self.is_ind(
                                seg_list[j + 1]):
                            backward_merge()
                        else:
                            forward_merge()
                    # 如果后一个是地区词，则与前面的合并
                    elif j < len(seg_list) - 1 and self.is_loc(seg_list[j + 1]):
                        # 但是如果前一个是大于等于4个字的，则后面的地区词可能判断错误，与后一个合并
                        if j > 0 and len(seg_list[j - 1]) >= 4:
                            backward_merge()

                        # 如果该词是人工标注需要向后合并的，且前面一个词是行业词，则地区可能判断出错，与后一个合并
                        elif self.one_word_pos.get(seg_list[j], 0) == 1 and j > 0 and self.is_ind(seg_list[j - 1]):
                            backward_merge()

                        else:
                            forward_merge()

                    elif self.one_word_pos.get(seg_list[j], 0) == 1 and j < len(seg_list) - 1:  # 如果是人工标注的需要往后合并的位置
                        backward_merge()

                    elif self.one_word_pos.get(seg_list[j], 0) == -1:  # 如果是人工标注的需要往前合并的位置
                        forward_merge()

                    else:  # 如果都不满足
                        adjusted_seg.append(seg_list[j])
            # 不是一个字的也看向哪儿合并
            elif self.one_word_pos.get(seg_list[j], 0) == 1 and j < len(seg_list) - 1:  # 如果是人工标注的需要往后合并的位置
                backward_merge()

            elif self.one_word_pos.get(seg_list[j], 0) == -1 and j > 0:  # 如果是人工标注的需要往前合并的位置
                forward_merge()

            else:
                adjusted_seg.append(seg_list[j])
                # print('adjusted_seg13',adjusted_seg)
        # 拆分策略(大于等于4个字的)
        adjusted_seg_split = []
        for seg in adjusted_seg:
            # 从前往后看地区
            reverse = False  # 是否需要从后往前看
            skip_while = False  # 是否跳出while循环
            while len(seg) >= 4 and not seg.encode('utf8').isalpha() and not skip_while:  # 如果大于4个字，且是中文，则做拆分处理
                # TODO:注意这里判断是否是英文的办法，在python2环境中需要修改，不兼容
                # n_gram
                for n in range(len(seg), 1, -1):  # 最低2元语法，最高为 字数-2 元语法（至少保证剩余两个字）
                    if n == len(seg) - 1 and (
                                    seg[-1] not in self.company_suffix or (
                                            seg[-1] in self.company_suffix and self.is_ind(seg[-2:]))):
                        # 这样拆分出来剩下的是一个字,如果这个字不是属于公司后缀，或者属于后缀，但是与前面一个词相连为行业词（说明不可分开）则跳过
                        continue
                    if self.is_loc(seg[:n]):
                        adjusted_seg_split.append(seg[:n])
                        seg = seg[n:]
                        break  # 一旦任何一次拆分成功，则立马停止剩下的几元语法的尝试，直接对剩余的词做处理
                    elif n == 2:  # 如果2元拆分再失败，则表示从前往后看无法拆分，需要从后往前看，并且跳出while循环
                        reverse = True
                        skip_while = True
                        break
            if not reverse and seg:  # 从while循环跳出来的，不足4字大于0字的，直接放入
                adjusted_seg_split.append(seg)

            if reverse:  # 从后往前看行业
                reverse_list = []  # 从后往前看的寄存器，用于储存拆分出来的词，由于是倒序看的，所以最后要反序
                skip_while = False
                while len(seg) >= 4 and not skip_while:
                    # n_gram
                    for n in range(len(seg) - 2, 1, -1):  # 最低2元语法，最高为 字数-2 元语法（至少保证剩余两个字）
                        if self.is_ind(seg[-n:]):
                            reverse_list.append(seg[-n:])
                            seg = seg[:-n]
                            break  # 一旦任何一次拆分成功，则立马停止剩下的几元语法的尝试，直接对剩余的词做处理
                        elif n == 2:  # 如果2元拆分再失败，则完全失败，跳出while循环
                            skip_while = True
                            break

                reverse_list.append(seg)
                reverse_list.reverse()
                adjusted_seg_split.extend(reverse_list)

        return adjusted_seg_split

    def company_split_tier(self, company):

        company = clean_punc(company)
        ordinal_num_set = set(['一', '二', '三', '四', '五', '六', '七', '八', '九', '十'])

        # 先清洗tag为0的词语
        for suffix, tier in self.company_suffix.items():
            if tier == '0':  # 先清洗tag为0的词语
                company = re.sub(suffix, '', company)

        raw_company = company
        for suffix, tier in self.company_suffix.items():
            tier = int(tier)

            if tier > 0 and len(suffix) > 1:
                # 对子公司的特殊处理
                if suffix != '子公司':
                    company = re.sub(suffix, '#' * len(suffix), company)  # 将需要分解的节点统一变形为#
                else:
                    company = list(company)
                    for i in range(len(company)):
                        if i < len(company) - 2:
                            if company[i:i + 3] == ['子', '公', '司'] and i > 0:
                                ind_check = company[i - 1] + '子'
                                if self.is_ind(ind_check):  # 如果前面的字加子是一个行业词，则子公司属于误判
                                    company[i + 1:i + 3] == ['#', '#']
                                else:
                                    company[i:i + 3] == ['#', '#', '#']

                    company = ''.join(company)
            elif len(suffix) == 1:  # 长度为1的尾缀，如果处于尾部，则可直接分拆；如果在中间出现，则需要观察前面的词的情况
                # 处理尾部情况
                if company.endswith(suffix):
                    company = company[:-len(suffix)] + '#' * len(suffix)
                # 如果在中部出现
                if suffix in company[:-len(suffix)]:
                    # 需要分词
                    temp_company = ''
                    cut_list = self.cut_adjust(list(jieba.cut(company)))
                    for i in range(len(cut_list)):

                        if i > 0:
                            former_word = cut_list[i - 1]
                        else:
                            former_word = ''
                        company_seg = cut_list[i]
                        if i < len(cut_list) - 1:
                            after_word = cut_list[i + 1]
                        else:
                            after_word = ''

                        if company_seg == suffix:  # 直接被分词分出来了，可直接分拆
                            temp_company += '#' * len(suffix)

                        # 出现在词语中，没有被单独分离出来，则需要判断其位置
                        # 直接出现在尾部，则判断其之前的词是否单独成词
                        elif company_seg.endswith(suffix):
                            # 前面是行业词，如包子店，饺子店;前面是地区词，如浦东店;前面是序数词，如一店、二部、三局;均可拆分
                            front_part = company_seg[:-len(suffix)]
                            if self.is_ind(front_part) or self.is_loc(front_part) or front_part[-1] in ordinal_num_set:
                                temp_company += front_part + '#' * len(suffix)
                            else:
                                temp_company += company_seg

                        # 直接出现在头部，可能误与后面合并，判断之前的词是否成词
                        elif company_seg.startswith(suffix):
                            #                             print('company_seg',company_seg)
                            if not former_word:  # 前面没有词
                                temp_company += company_seg
                            # 前面是行业词，如包子店，饺子店;前面是地区词，如浦东店;前面是序数词，如一店、二部、三局;且拆分之后，剩余的部分依然成词，则可拆分
                            elif (self.is_ind(former_word) or self.is_loc(former_word) or former_word[
                                -1] in ordinal_num_set) and not self.is_vac(company_seg):
                                temp_company += '#' * len(suffix) + company_seg[len(suffix):]
                            else:
                                temp_company += company_seg
                        else:
                            temp_company += company_seg
                    company = temp_company
                    #         print(company,raw_company)
        # 头部不能直接出现后缀，如果出现，则视为主体：
        for i in range(len(company)):
            if company[i] == '#':
                company = company[:i] + '*' + company[i + 1:]  # 替换成*避免被分组
            else:
                break
        # 根据打的位置标记，进行拆分：
        from itertools import groupby
        comp_seg = [''.join([i[0] for i in g]) for _, g in groupby(zip(raw_company, company), lambda s: s[1] == '#')]
        # 如果拆分出来是奇数，则末尾补空
        if len(comp_seg) % 2 == 1:
            comp_seg.append('')

        def suffix_split(word):
            # 对一个连续后缀进行后缀拆分
            split_result = []
            max_loop_num = len(word)
            loop_num = -1
            while len(word) > 0 and loop_num < max_loop_num:
                loop_num += 1
                for i in range(len(word)):
                    pre = word[:i]
                    suf = word[i:]
                    if suf in self.company_suffix.keys():
                        split_result.append(suf)
                        word = pre
                        break

            split_result.reverse()
            return split_result

        comp_seg_full = []  # 完整拆分的列表
        for i in range(0, len(comp_seg), 2):
            main_part = comp_seg[i]
            suf_group = comp_seg[i + 1]
            suf_group_splited_list = suffix_split(suf_group)
            comp_seg_full.append(main_part)
            comp_seg_full.extend(suf_group_splited_list)
        # 对拆分的结果，标注等级
        comp_tier = [int(self.company_suffix.get(i, 100)) for i in comp_seg_full]  # 主体标注为100

        # 对拆分进行调整：
        # 如果等级为5，且后面有小于自己的等级，则该等级为5的切割不该存在
        for i in range(len(comp_tier)):
            if comp_tier[i] != 100 and i < len(comp_tier) - 1 and comp_tier[i] > min(
                    comp_tier[i + 1:]):  # 主体直接略过，对于后缀，如果后面有小于自己的等级的，自己直接归为主体
                comp_tier[i] = 100

        # 对于连续后缀：如果后缀的头两个后缀，都是一个字，则，后缀的第一个字应当属于主体（如旅行/社/公司）
        for i in range(len(comp_tier)):
            if 0 < i < len(comp_tier) - 1 and comp_tier[i - 1] == 100 and comp_tier[i] != 100 and comp_tier[
                        i + 1] != 100 and len(comp_seg_full[i]) == 1 and len(comp_seg_full[i + 1]) == 1:
                # 前一个是主体，自己是长度为1的后缀，下一个也是长度为1的后缀
                comp_tier[i] = 100

        # 【特殊规则】：对于“门市”可能组成的地区词进行过滤
        menshi_loc_filter = set(['海门市', '厦门市', '江门市', '荆门市', '天门市', '玉门市'])
        for i in range(len(comp_tier)):
            if i > 0 and comp_seg_full[i] == '门市':
                if comp_seg_full[i - 1][-1] + comp_seg_full[i] in menshi_loc_filter:
                    comp_tier[i] = 100

        # 每个等级只能呆一个组织，如果后面与前面的等级有重复的，则取前面最大等级+1(tier=-1的由于是删除项，除外)：
        tier_set = set([0])
        for i in range(len(comp_tier)):
            if comp_tier[i] in tier_set:
                comp_tier[i] = max(tier_set) + 1
            if comp_tier[i] != 100:
                tier_set.add(comp_tier[i])

        # 进行等级的第二次调整：之前可能会将等级加到大于5，凡是大于5的，大家进行等级平移
        diff = max(0, max(tier_set) - 5)  # 小于0则不平移
        if diff:
            for i in range(len(comp_tier)):
                if comp_tier[i] != 100:  # 主体不平移
                    comp_tier[i] = comp_tier[i] - diff

        # 最后根据等级进行拆分，连续的作为一组
        from itertools import groupby
        comp_seg_tier = [list(g) for label, g in groupby(zip(comp_seg_full, comp_tier), lambda s: s[1] == 100)]
        comp_seg_tier = [(''.join(i), min(j)) for i, j in [zip(*i) for i in comp_seg_tier]]

        # 组数据：（主体、后缀、等级）
        comp_info = []
        for i in range(0, math.ceil(len(comp_seg_tier) / 2) * 2, 2):
            main_part = comp_seg_tier[i][0]
            if i < len(comp_seg_tier) - 1:
                suf = comp_seg_tier[i + 1][0]
                tier = comp_seg_tier[i + 1][1]
            else:
                suf = ''
                tier = 5
            comp_info.append([main_part, suf, tier])

        return comp_info

    # 对公司进行彻底分词：先整体拆分为段，再进行调整分词，再拆分为组织主体分离
    def absolutely_cut(self, company):
        '''
        强行斩断可能引起不可预料的分词效果，比如餐厅的餐，和前面的词合并了
        '''
        all_seg = []
        company_split_result = self.company_split_tier(company)
        for main_part, org, tier in company_split_result:
            seg_list = self.cut_adjust(list(jieba.cut(main_part + org)))
            '''
            分词后没有再度分拆出来的，有两种情况:
            一种是，后缀是店，分词是商店这种，分词后字数多了的；
            一种是后缀是连锁有限公司，分词后是连锁/有限公司，分词后字数少了的。通常分词字数少了不用管，但是也可能出现部分词被合并到前面去的情况，
            比如第一分公司，变成了第一分/公司，这样就是有问题的了
            '''
            # 对分词结果进行调整
            if not org:
                pass
            elif seg_list and len(seg_list[-1]) > len(org):  # 分词后字数多了的，则强行分拆即可
                seg_list[-1] = seg_list[-1][:-len(org)]
                seg_list.append(org)
            elif seg_list and len(seg_list[-1]) < len(org):  # 分词后字数变少了的，则不断往前比，如果恰好完全符合，则没问题
                res_len = len(org)
                suf_index = 1
                while res_len > 0:  # 只要没减完，就持续光标前移，继续减
                    res_len = res_len - len(seg_list[-suf_index])
                    suf_index += 1
                # 如果刚好减完，则未出现合并现象（如后缀是分公司，但是合并成了 第一分/公司）
                # 如果减完停止是负数，则说明有后缀被合并，强行拆分
                if res_len < 0:
                    suf_index -= 1
                    # 此时abs(res_len)表示多出的字
                    res_word = seg_list[-suf_index][abs(res_len):]
                    seg_list[-suf_index] = seg_list[-suf_index][:abs(res_len)]
                    if res_word in self.company_suffix:  # 如果分离出来的词在后缀中，则直接插入
                        seg_list.insert(-suf_index + 1, res_word)
                    else:  # 如果不在后缀，则与下一个词合并
                        seg_list[-suf_index + 1] = res_word + seg_list[-suf_index + 1]
            all_seg.extend(seg_list)
        return all_seg, company_split_result

    def cluster_top_simi(self, cluster_vec, single_vec, topn=5):
        '''
        计算一个词处于某一类的概率
        '''
        simi_vec = np.dot(cluster_vec, single_vec)
        topn_simi = simi_vec[matutils.argsort(simi_vec, topn=topn, reverse=True)]
        return topn_simi.mean(0)

    def name_probability(self, word):
        try:
            word_vec = self.model[word]
        except:
            return 0
        return self.cluster_top_simi(self.name_vec, word_vec)

    def ind_probability(self, word):
        try:
            word_vec = self.model[word]
        except:
            return 0
        return self.cluster_top_simi(self.ind_vec, word_vec)

    def loc_probability(self, word):
        try:
            word_vec = self.model[word]
        except:
            return 0
        return self.cluster_top_simi(self.loc_vec, word_vec)