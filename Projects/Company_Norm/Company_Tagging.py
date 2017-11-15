'''
本文件用于将一个公司正式彻底进行拆解，并标注响应位置信息
'''

from util import Company_Cut, find_range
import numpy as np

org_cut = Company_Cut()


def company_group(company):
    '''
    对公司先划分组别和组内的分词结果
    '''
    all_seg, split_result = org_cut.absolutely_cut(company)
    group_dict = []
    for main_part, org, tier in split_result:
        main_part_seg = []
        while len(''.join(main_part_seg)) < len(main_part):  # 没有加够足够的字就继续加
            main_part_seg.append(all_seg[0])
            del all_seg[0]
        org_seg = []
        while len(''.join(org_seg)) < len(org):  # 没有加够足够的字就继续加
            org_seg.append(all_seg[0])
            del all_seg[0]
        if len(''.join(main_part_seg)) > len(main_part) or len(''.join(org_seg)) > len(org):
            raise Exception('出现了不可预知的词长匹配错误')

        # 对银行、大学等进行特殊处理：（为了拆分方便，把银行整体算作了公司尾缀，最后加回来）
        particular_suffix = {'银行', '大学', '中学', '小学', '储蓄所', '分理处', '营业部', '经营部', '连锁店', '经营点', '加盟店'}
        for particular_suf in particular_suffix:
            if org.startswith(particular_suf):
                main_part_seg.append(org[:2])
                org_cut.vac_tag[org[:2]] = 'ind'
                org = org[2:]

        group_dict.append({'main_seg': main_part_seg, 'suffix': org, 'tier': tier})
    return group_dict


def company_split_tag(company):
    mode = {('loc', 'name', 'ind'), ('name', 'loc', 'ind'), ('name', 'ind', 'loc'), ('name', 'ind'), ('loc', 'name'),
            ('name', 'loc'), ('name',)}
    group_dict = company_group(company)

    def tag_position(tag_array, tag, pos=0):
        '''
        给出tag在array中的index
        tag:需要寻找位置的tag
        pos:需要给出的位置，0表示第一个，-1表示最后一个
        '''
        pos_index = np.where(tag_array == tag)[0]
        if pos_index.shape[0]:
            return pos_index[pos]
        else:
            return None

    def generate_main_former_after(iter_list, index, times=1):
        '''
        本程序用于寻找一个可迭代对象某元素前后的元素
        times：向前向后寻找几个,默认为1。最多为2
        '''
        if times > 2 or times < 1:
            raise Exception('只能向前向后寻找1-2层')
        main = iter_list[index]
        if index < 1:
            former = ''
        else:
            former = iter_list[index - 1]
        if index > len(iter_list) - 2:
            after = ''
        else:
            after = iter_list[index + 1]
        if times == 2:
            if index < 2:
                former_former = ''
            else:
                former_former = iter_list[index - 2]
            if index > len(iter_list) - 3:
                after_after = ''
            else:
                after_after = iter_list[index + 2]
        else:
            former_former = ''
            after_after = ''
        return main, former, after, former_former, after_after

    def generate_surround_list(seg_tag_list):
        '''
        · 生成模式包围列表
        · 至少3个元素谈包围才有意义
        · 边缘处理：两侧越界处，均处理成边界为轴，旁侧镜像映射的那个值。（如[a,b,c,d],a的左侧处理为b，d的右侧处理为c）
        · 几个定义：
            1. 不包围：某tag两侧的tag不为同一个tag，记号记为'none'。如[a,b,c],b处于不被包围
            2. 半包围：某tag两侧的tag为同一个tag，但是，继续往两侧扩散，tag发生变更。即包围该tag的墙只有两堵，记号为'half'。如[a,b,a]，b处于被a半包围
            3. 全包围：某tag两侧的tag为同一个tag，且至少一个方向，包围的tag数量大于1，记号记为'all'。如[a,b,a,a]，b处于被a全包围
        '''
        # 判断列表长度:小于等于2的，直接返回不包围
        if len(seg_tag_list) <= 2:
            return ['none'] * len(seg_tag_list)

        # 生成模式包围列表
        surround_list = []
        for i in range(len(seg_tag_list)):
            # 处理边界情况
            main, former, after, former_former, after_after = generate_main_former_after(seg_tag_list, i, 2)
            # 开始判断包围情况 abc abb bbababb aab
            if (not former and after_after != after) or (not after and former_former != former) or (
                    former and after and former != after) or main == after:  # 前后不是同一部队（abc、abac类型），或者前后部队一致，但是是自己的友军，都算没有被包围
                surround_list.append('none')
            elif after_after == after == former or former_former == former == after:  # 前后一个方向继续扩散包围（abaa类型）
                surround_list.append('all')
            else:  # 如果只是前后被包围，则这一组均标记为half（abac类型）
                surround_list.append('half')  # 本组标记为half
        # 半包围调整（abac类型）
        skip = 0
        for i in range(len(surround_list)):
            if skip:
                skip -= 1
                continue
            main, former, after, former_former, after_after = generate_main_former_after(surround_list, i, 1)
            if main == 'half':
                if former == 'none':
                    surround_list[i - 1] = 'half'
                if after == 'none':
                    surround_list[i + 1] = 'half'
                    skip = 1  # 防止继续扩散

        return surround_list

    def generate_group_pattern(to_group_list):
        return tuple([i for i, j in groupby(to_group_list)])

    def generate_tag_count(seg_tag_list):
        tag_count = {}
        for seg_tag in seg_tag_list:
            tag_count[seg_tag] = tag_count.get(seg_tag, 0) + 1
        return tag_count

    def all_possible_cut(text):
        cut_result = []
        for length in range(2, len(text)):
            for index in range(len(text) + 1 - length):
                cut_result.append(text[index:index + length])
        return cut_result

    company_seg_result = []
    for company_tier_info in group_dict:  # company_tier_info:{'main_seg': main_part_seg, 'suffix': org, 'tier': tier}
        #         print(company_tier_info)
        seg_tag_list = []  # tag按照顺序给出的tag列表
        main_seg_list = company_tier_info['main_seg']
        for main_seg in main_seg_list:
            seg_tag = org_cut.vac_tag.get(main_seg, 'none')
            if seg_tag == 'none' and org_cut.is_loc(main_seg):  # 如果明确是地区词，但是没有再词库中，则先归到地区词
                seg_tag = 'loc'
            seg_tag_list.append(seg_tag)
        seg_tag_list = np.array(seg_tag_list, dtype='<U4')

        # 尾部一个字被判定为name的，均替换为ind
        if len(main_seg_list[-1]) == 1 and seg_tag_list[-1] == 'name':
            seg_tag_list[-1] = 'ind'
        # org_cut.ind_vec=np.append(org_cut.ind_vec,[org_cut.model[main_seg_list[-1]]],axis=0)

        from itertools import groupby
        group_pattern = generate_group_pattern(seg_tag_list)
        group_count = generate_tag_count(group_pattern)  # 计算各自分成了几组
        max_count = max(group_count.values())
        # print('处理前seg_tag_list', seg_tag_list)
        # 对于某个部分不连续，被拆分成了多组的（如loc,ind,loc），对tag进行规则调整：
        #         while_loop_count=1
        while max_count > 1:  # 注意这里处理不了'none'的情况，比如loc,loc,name,none,ind,ind，max_count==1
            # 记录世界初始格局
            raw_group_pattern = generate_group_pattern(seg_tag_list)
            # print('世界初始格局', raw_group_pattern)
            # 先确定需要叛变哪些tag（count>=2的均有可能需要叛变）
            betray_tag = set([i for i, j in group_count.items() if j >= 2])
            # print('需要叛变的tag', betray_tag)
            # 再确定这些tag可能的叛变方向
            betray_target_possible = []
            # 如果某个tag不在原有tag列表中，则其可以是任何一个叛军可能的叛变方向
            spare_target = list({'name', 'ind', 'loc'} - set(seg_tag_list))
            for i in range(len(seg_tag_list)):
                main, former, after, former_former, after_after = generate_main_former_after(seg_tag_list, i, 1)
                # 如果不在需要叛变的tag列表中，叛变方向为'none'。但是如果两侧都是需要叛变的tag，其受叛军影响，其也可能需要叛变
                # 但是需要注意的是，夹在叛军中间的只能向叛军叛变，而叛军可以向另一个不存在的部队进行叛变
                # 如：loc loc ind loc name ind，你不知道是因为ind出错，导致loc被分开，还是因为loc出错导致分开
                if main not in betray_tag and (former not in betray_tag or after not in betray_tag):
                    betray_target_possible.append(['none'])
                else:
                    # 如果周围有友军，则不叛变
                    if main == former or main == after:
                        betray_target_possible.append(['none'])
                    # 开始尝试将其前后相邻的以及未出现的部队作为叛变目标，可能有多个叛变目标
                    else:
                        temp_target_list = []
                        # 如果其前后都是叛军，则只能向叛军叛变
                        if former in betray_tag and after in betray_tag:
                            betray_target_allow = [former, after]
                        # 否则还可以向没有出现的军队进行叛变
                        else:
                            betray_target_allow = [former, after] + spare_target
                        for target in betray_target_allow:
                            # 如果不存在，则跳过
                            if not target:
                                continue
                            # 如果叛变之后回到正常状态，但是ind在name之前，则不朝这个方向叛变（叛变禁忌1）
                            try_seg_tag_list = np.array(seg_tag_list)
                            try_seg_tag_list[i] = target
                            try_group_pattern = generate_group_pattern(try_seg_tag_list)
                            try_group_count = generate_tag_count(try_group_pattern)
                            try_max_count = max(try_group_count.values())
                            if try_max_count <= 1 and 'name' in try_seg_tag_list and 'ind' in try_seg_tag_list and tag_position(
                                    try_seg_tag_list, 'name', 0) > tag_position(try_seg_tag_list, 'ind', 0):
                                continue
                            temp_target_list.append(target)
                        if temp_target_list:
                            betray_target_possible.append(temp_target_list)
                        else:
                            betray_target_possible.append(['none'])
            # print('可能叛变的方向', betray_target_possible)
            # 再计算叛变的概率
            # 先生成包围列表：主要为处理被包围情况下，叛变概率为1
            # print('before surround', seg_tag_list)
            surround_list = generate_surround_list(seg_tag_list)
            # print('surround_list', surround_list)
            betray_rate_list = []
            betray_target_finnal = []  # 最终的叛变方向
            for i in range(len(betray_target_possible)):
                # 全包围情况下，叛变概率为1，叛变方向也肯定只有一个
                #                 if surround_list[i]=='all':
                # 如果是第一次循环，先替换中间的all，先不处理边界处的all。如果处理完没有符合退出条件，再处理边界处的all。
                # 如：[修文县 富民 钧发 果蔬 种植 农民 专业 合作社],初始tag_list为[loc name name ind ind name ind ind ind]
                #                     if (i == 0 or i == len(surround_list) - 1) and while_loop_count == 1:
                #                         betray_rate=0
                #                     else:
                #                     betray_rate=1
                #                     betray_rate_list.append(betray_rate)
                #                     betray_target_finnal.append(betray_target_possible[i][0])
                #                 else:
                betray_rate_dict = {}
                for target in betray_target_possible[i]:
                    try:
                        rate_cal_func = getattr(org_cut, target + '_probability')
                    except:
                        rate_cal_func = lambda x: 0  # 朝'none'叛变的概率为0
                    betray_rate_dict[target] = rate_cal_func(main_seg_list[i])  # 叛变概率
                    # 【特殊规则：如果是最后一个地区词需要叛变，叛变概率乘2】
                    if i == len(betray_target_possible) - 1 and len(main_seg_list[i]) >= 4 and seg_tag_list[i] == 'loc':
                        betray_rate_dict[target] = betray_rate_dict[target] * 2
                    # 【特殊规则：全包围情况下，叛变概率乘3】
                    if surround_list[i] == 'all':
                        betray_rate_dict[target] = betray_rate_dict[target] * 3

                # 在可能的叛变方向中选择叛变概率最大的作为叛变方向。如果最大的概率为0，表示不叛变
                betray_rate = max(betray_rate_dict.values())
                betray_rate_list.append(betray_rate)
                if betray_rate == 0:
                    betray_target_finnal.append('none')
                else:
                    betray_target_finnal.append(max(betray_rate_dict, key=betray_rate_dict.get))
            # print('叛变概率', betray_rate_list)
            # print('最终叛变方向', betray_target_finnal)
            # 正式开始叛变：每次只叛变一次，叛变概率最高的进行叛变。（因为每次叛变后将会影响世界格局，世界格局需要重新分配和计算）
            betray_rate_list = np.array(betray_rate_list)
            max_betray_rate = betray_rate_list.max()
            betray_index = betray_rate_list.argmax()  # 需要进行叛变的元素索引，如果有多个最大值，返回第一个
            # 如果最大的叛变概率都是0，则表示世界和平，不进行叛变，跳出循环
            if max_betray_rate == 0:
                break
            # 执行叛变
            # print('执行叛变', betray_index, betray_target_finnal[betray_index])
            seg_tag_list[betray_index] = betray_target_finnal[betray_index]
            # 重新计算世界格局
            group_pattern = generate_group_pattern(seg_tag_list)
            group_count = generate_tag_count(group_pattern)  # 计算各自分成了几组
            max_count = max(group_count.values())
            #             while_loop_count+=1
            # print('叛变后世界格局', group_pattern)
            # print('叛变后seg_tag_list', seg_tag_list)
            # 如果叛变后世界格局未发生变化，则停止叛变
            if group_pattern == raw_group_pattern:
                break

        # 处理none的情况
        if 'none' in group_pattern:
            # 如果没有字号，且只有一个none（连续的算一个）的，则把none替换为字号
            if 'name' not in group_pattern and group_count['none'] == 1:
                seg_tag_list[seg_tag_list == 'none'] = 'name'
            else:
                '''
                否则采取如下策略：
                1. 两个字的，如果name不在原有列表中，则视为字号。如果name已经存在，则：如果none在name旁边，变成name；否则随上家或下家，优先上家
                2. 三个字以上的拆分成所有可能的两个字到N-1个字的组合，取平均，看更接近旁边的哪个。怎么取也找不到的，算字号
                3. 中英夹杂，无法区分英文的是字号

                '''
                for i in range(len(seg_tag_list)):
                    main, former, after, former_former, after_after = generate_main_former_after(seg_tag_list, i, 1)
                    if main != 'none':
                        continue
                    if len(main_seg_list[i]) <= 2:
                        if 'name' not in seg_tag_list:  # 如果name不在原有列表中，则视为字号
                            seg_tag_list[i] = 'name'
                        else:
                            if former == 'name' or after == 'name':  # name已经在原有列表，但是旁侧有name，可以相连，则视为字号
                                seg_tag_list[i] = 'name'
                            elif former:  # 旁边没有name，则优先随上家
                                seg_tag_list[i] = former
                            else:  # 如果上家没有，则随下家
                                seg_tag_list[i] = after
                    elif main_seg_list[i].encode('utf8').isalpha():
                        # 自己是英文，整体不是全英文(暂时去掉，整体是英文也没办法)
                        seg_tag_list[i] = 'name'
                    else:
                        possible_cut_list = all_possible_cut(main_seg_list[i])
                        max_target_prob = {}
                        for target_tag in [former, after]:
                            target_prob_list = []
                            try:
                                rate_cal_func = getattr(org_cut, target_tag + '_probability')
                            except:
                                rate_cal_func = lambda x: 0
                            for possible_cut in possible_cut_list:
                                target_prob_list.append(rate_cal_func(possible_cut))
                            max_target_prob[target_tag] = max(target_prob_list)
                        if max_target_prob[former] > max_target_prob[after]:
                            seg_tag_list[i] = former
                        elif max_target_prob[former] < max_target_prob[after]:
                            seg_tag_list[i] = after
                        else:
                            seg_tag_list[i] = 'name'

        # 更新group_pattern信息和tag_count信息
        group_pattern = generate_group_pattern(seg_tag_list)
        tag_count = generate_tag_count(seg_tag_list)
        # 先处理缺失字号的场景
        if tag_count.get('name', 0) == 0:
            # 缺失字号场景：分支机构，如中国银行上海五角场分行，上海五角场都是地区，name应该是中国，行业应该是银行
            # 下面执行字号继承策略
            if company_seg_result:
                # 如果之前已经有结果了分层的结果了，则直接可以继承之前的字号
                parent_level_org_name = ''.join(company_seg_result[-1].get('name', []))
                # 直接继承，把字号直接插入列表首部
                main_seg_list.insert(0, parent_level_org_name)
                seg_tag_list = np.insert(seg_tag_list, 0, 'name')
                group_pattern = generate_group_pattern(seg_tag_list)  # 重新计算group_pattern
            if group_pattern in {('loc', 'ind'), ('ind',), ('loc',)}:  # 缺失字号场景一：误将字号判断为loc或者ind
                last_loc_index = tag_position(seg_tag_list, 'loc', -1)  # 最后一个地区的index
                first_ind_index = tag_position(seg_tag_list, 'ind', 0)  # 第一个行业的index
                if tag_count.get('loc', 0) > tag_count.get('ind', 0):  # 如果地区的tag比行业的tag多，则最后一个地区变更为字号
                    seg_tag_list[last_loc_index] = 'name'
                elif tag_count.get('loc', 0) < tag_count.get('ind', 0):  # 如果地区的tag比行业的tag少，则第一个行业变更为字号
                    seg_tag_list[first_ind_index] = 'name'
                else:  # 如果个数相等
                    # 如果地区和行业均只有一个，则将其联合起来统一作为字号（如北京银行、中国邮政）
                    if tag_count['loc'] == tag_count['ind'] == 1:
                        main_seg_list.insert(1, main_seg_list[0] + main_seg_list[1])  # 把新生成的字号加入到第二个位置
                        seg_tag_list = np.insert(seg_tag_list, 1, 'name')
                    # 否则看谁是name的概率高
                    else:
                        last_loc = main_seg_list[last_loc_index]
                        first_ind = main_seg_list[first_ind_index]
                        last_loc_name_prob = org_cut.name_probability(last_loc)
                        first_ind_name_prob = org_cut.name_probability(first_ind)
                        if last_loc_name_prob < first_ind_name_prob:  # 第一个行业的字号概率更高
                            seg_tag_list[first_ind_index] = 'name'
                        else:
                            seg_tag_list[last_loc_index] = 'name'  # 如果概率相等（which概率很低），随便将loc变为字号
            if 'none' in seg_tag_list:  # 缺失字号场景二：没字号的，有none的，则把none改为字号
                seg_tag_list[seg_tag_list == 'none'] = 'name'
        # 行业继承
        if tag_count.get('ind', 0) == 0:
            # 缺失行业场景：分支机构，如中国银行上海五角场分行，上海五角场都是地区，name应该是中国，行业应该是银行
            # 下面执行行业继承策略
            if company_seg_result:
                # 如果之前已经有结果了分层的结果了，则直接可以继承之前的字号
                parent_level_org_name = ''.join(company_seg_result[-1].get('ind', []))
                # 直接继承，把行业直接插入列表尾部
                main_seg_list.insert(-1, parent_level_org_name)
                seg_tag_list = np.insert(seg_tag_list, -1, 'ind')
        # 输出结果
        for seg, tag in zip(main_seg_list, seg_tag_list):
            company_tier_info.setdefault(tag, []).append(seg)
        company_seg_result.append(company_tier_info)

        # 隐含分割
        # 地区合并和判断和归一化

        # 先观察一下出错的场景都有哪些，再来研究怎么搞
        # 更新group_pattern信息
        group_pattern = generate_group_pattern(seg_tag_list)
        if group_pattern in mode:
            # print('right_seg', list(zip(company_tier_info['main_seg'], seg_tag_list)))
            pass
        else:
            print(company)
            print('not normal mode', list(zip(company_tier_info['main_seg'], seg_tag_list)))
            if tag_count.get('name', 0) > 1:
                print('name_count_larger_than_1', list(zip(company_tier_info['main_seg'], seg_tag_list)))
            if tag_count.get('name', 0) == 0:
                print('no_name', list(zip(company_tier_info['main_seg'], seg_tag_list)))
            if 'name' in seg_tag_list and 'ind' in seg_tag_list and tag_position(seg_tag_list, 'name',
                                                                                 0) > tag_position(seg_tag_list, 'ind',
                                                                                                   0):
                print('ind_before_name', list(zip(company_tier_info['main_seg'], seg_tag_list)))
            if len(list(groupby(seg_tag_list))) > 3:
                print('more_than_3_parts', list(zip(company_tier_info['main_seg'], seg_tag_list)))

            print(('{0:-^50}').format(''))
    return company_seg_result


if __name__ == '__main__':
    print(company_split_tag('中国移动通信集团有限公司'))
