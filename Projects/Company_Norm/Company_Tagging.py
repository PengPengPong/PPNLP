'''
本文件用于将一个公司正式彻底进行拆解，并标注响应位置信息
'''


from util import Company_Cut,find_range

org_cut=Company_Cut()


def company_group(company,company_suffix):
    '''
    对公司先划分组别和组内的分词结果
    '''
    all_seg,split_result=org_cut.absolutely_cut(company,company_suffix)
    group_dict=[]
    for main_part,org,tier in split_result:
        main_part_seg=[]
        while len(''.join(main_part_seg))<main_part: # 没有加够足够的字就继续加
            main_part_seg.append(all_seg[0])
            del all_seg[0]
        if len(''.join(main_part_seg))>main_part:
            raise Exception('出现了不可预知的词长匹配错误')
        group_dict.append({'main_seg':main_part_seg,'suffix':org,'tier':tier})
    return group_dict

def company_split_tag(company,company_suffix):
    mode=set([('loc','name','ind'),('name','loc','ind'),('name','ind','loc'),('name','ind'),('loc','name'),('name','loc'),('name')])
    group_dict=company_group(company,company_suffix)
    for company_tier_info in group_dict:
        seg_tag_list=[] # tag按照顺序给出的tag列表
        tag_count={}
        for main_seg in company_tier_info['main_seg']:
            seg_tag=org_cut.vac_tag.get(main_seg,'name')
            tag_count[seg_tag]=tag_count.get(tag_count,0)+1
            seg_tag_list.append(seg_tag)


        # 下面对tag进行规则调整：
        # 1. 字号词只能有一个，相连的算一个，多于的全算行业


        # 先观察一下出错的场景都有哪些，再来研究怎么搞
        from itertools import groupby
        if tag_count.get('name',0)>1:
            print('name_count_larger_than_1',list(zip(company_tier_info['main_seg'],seg_tag)))
        if tag_count.get('name',0)==0:
            print('no_name',list(zip(company_tier_info['main_seg'],seg_tag)))
        if seg_tag_list.index('name')>seg_tag_list.index('ind'):
            print('ind_before_name',list(zip(company_tier_info['main_seg'],seg_tag)))
        if len(list(groupby(seg_tag_list)))>3:
            print('more_than_3_parts',list(zip(company_tier_info['main_seg'],seg_tag)))
        group_pattern=[i for i,j in groupby(seg_tag_list)]
        if group_pattern not in mode:
            print('not normal mode',list(zip(company_tier_info['main_seg'],seg_tag)))
