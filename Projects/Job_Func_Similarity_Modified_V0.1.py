# 职位为某个职能的概率
# 预先算好，之后对每个猎头就不用重复计算，直接提取即可

def Job_Func_Similarity(JobTitle, JobDuty, FuncList, Percentage=0.2):
    '''
    Intro:
        · 预先计算一个职位和每个职能的匹配度
    Args:
        · JobTitle:职位
        · JobDuty:职位描述
        · Subscription:职能列表
    Return:
        · 职能相似度矩阵：行为职位的每个特征（职位名、关键词），列为职能，矩阵元素代表该特征为某个职能的概率
    '''

    #     高级管理岗位预处理：一些职衔词会被丢弃 # TODO:Modified
    for job in function_dict['高级管理']:
        if job == JobTitle[-len(job):]:
            JobTitle = JobTitle + '/CEO'
            break
    if '首席' in JobTitle:
        JobTitle = JobTitle + '/CEO'

    # 获取职位向量
    jobvac = Get_JobVac(JobTitle)  # Modified
    # 获取职位描述向量集合
    dutyvac = Get_DutyVac(JobDuty, Percentage)  # New

    # 如果职位向量为空
    if not jobvac:

        for i in ['助理', '文员', '助手']:
            if i in JobTitle:
                jobvac = Get_JobVac('文秘')
                break

    # 分别计算职位名和职位关键词落在每个职能的概率，这里不把职位名也作为关键词一起送入计算的原因是，这样方便之后调整职位名-职能相似度和关键词-职能相似度的权重

    # 计算职位名-职能相似度

    max_job_simi = []
    for func in FuncList:
        job_simi_subfunc = [0]
        for func_vec in function_vec[func]:
            for jobvac_seg in jobvac:
                try:
                    job_simi_subfunc.append(CosSimilarity(jobvac_seg, func_vec, PreNormalized=True))
                except:
                    continue
        max_job_simi.append(max(job_simi_subfunc))  # 职位名与某职能的相似度
    Job_Func_Matrix = pd.DataFrame([max_job_simi], index=['title'], columns=FuncList)

    # 计算关键词-职能相似度
    for keyword, dutyvac_seg in dutyvac:
        duty_simi_func = []
        for func in FuncList:
            duty_simi_subfunc = [0]
            for func_vec in function_vec[func]:
                try:
                    duty_simi_subfunc.append(
                        CosSimilarity(dutyvac_seg, func_vec, PreNormalized=True))  # 某关键词与所有订阅子职能的相关度
                except:
                    continue
            duty_simi_func.append(max(duty_simi_subfunc))
        Job_Func_Matrix = Job_Func_Matrix.append(pd.DataFrame([duty_simi_func], index=[keyword], columns=FuncList))

    # 对关键词的结果进行调整：由于取的是最大的，可能会由于一些偶尔的噪音，导致结果偏大。尝试去除与整体情况格格不入的最大值。目前是非常粗糙的办法。
    wordnum = Job_Func_Matrix.shape[0] - 1
    # 对关键词个数大于5的才做此调整
    if wordnum > 5:
        max_num = 2 if wordnum > 10 else 1
        for column, series in job_func_matrix.iloc[1:].iteritems():
            nlargest = series.nlargest(4)
            for i in range(max_num):
                try:
                    if nlargest[0] > 2 * nlargest[max_num] and nlargest[0] != series[0]:
                        job_func_matrix[column][series[series == nlargest[0]].index[0]] = 0
                except:
                    continue

    return Job_Func_Matrix


# 获取用户的订阅相似度
def Get_User_Subscription_Similarity(JobFuncMatrix, Subscription):
    Subscription = set(Subscription)
    subs_array = []
    for func in function_vec.keys():
        if func in Subscription:
            subs_array.append(1)
        else:
            subs_array.append(0)
    func_simi_matrix = (JobFuncMatrix * np.array(subs_array)).max(0)
    max_similarity = func_simi_matrix.max()
    max_func = func_simi_matrix.argmax()
    return (max_func, max_similarity)