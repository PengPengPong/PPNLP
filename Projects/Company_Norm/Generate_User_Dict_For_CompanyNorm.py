'''
本文件主要用于生成公司归一化所需要的用户字典
'''


def generate_user_dict():
    # 先将两个来源的地区字典合并
    location_set = set()
    for line in open('/Users/pp/pycharmprojects/nlp/locations.txt'):
        location_set.add(line.strip())
    for line in open('/Users/pp/pycharmprojects/Data/归一化标注/地区词表（含道路）.txt'):
        location_set.add(line.strip())

    # 进行部分地区的补全：给两个字的地区加上后缀
    # 导入loc_suffix：
    loc_suffix_set = set([])
    for line in open('/Users/pp/pycharmprojects/Data/归一化标注/loc_suffix.txt'):
        loc_suffix_set.add(line.strip())

    for loc in location_set.copy():
        if len(loc) == 2 and loc[-1] not in loc_suffix_set:
            for loc_suffix in loc_suffix_set:
                location_set.add(loc + loc_suffix)

    # 自带的地区词表中有部分是组织机构词，需要去除
    loc_suffix_to_be_del = set()
    for line in open('/Users/pp/pycharmprojects/Data/归一化标注/user_loc_suffix_to_be_del.txt', 'r'):
        # 从company_suffix拷贝而来，略作修改
        loc_suffix_to_be_del.add(line.strip())

    for loc in location_set.copy():
        for suf in loc_suffix_to_be_del:
            if len(suf) == 1:
                continue
            if loc[-len(suf):] == suf:
                location_set.remove(loc)
                break

    # 将完整的地区词表写入文件，待以后用
    with open('/Users/pp/pycharmprojects/Data/Companys/User_Loc_Dict_Full.txt', 'w') as f:
        for loc in location_set:
            f.write(loc + ' ' + str(1) + '\n')

    # 与之后统计出来的，单字和前后组合形成的固定搭配词表联合
    for line in open('/Users/pp/pycharmprojects/data/归一化标注/单字前后成词词典.txt', 'r'):
        location_set.add(line.strip())

    # 与人工筛选出来的行业词表联合
    for line in open('/Users/pp/pycharmprojects/data/归一化标注/行业词表_最终确认.txt', 'r'):
        location_set.add(line.strip())

    # 写入文件，完成
    with open('/Users/pp/pycharmprojects/Data/Companys/User_Dict_For_CompanyNorm.txt', 'w') as f:
        for loc in location_set:
            f.write(loc + ' ' + str(5) + '\n')
        # 与之后统计出来的，单字和前后组合形成的固定搭配词表联合
        for line in open('/Users/pp/pycharmprojects/data/归一化标注/debug过程用户补充词典.txt', 'r'):
            f.write(line.strip() + ' ' + str(1) + '\n')

if __name__ == '__main__':
    generate_user_dict()