import numpy as np


def normalizing_array(m):
    nm = np.zeros(m.shape)
    sum1 = 0
    for i in range(m.shape[1]):
        sum1 += m[0][i]
    for j in range(m.shape[1]):
        nm[0][j] = m[0][j] / sum1
    return nm


def nor_list(m):
    nor = []
    sum1 = sum(m)
    num = len(m)
    for i in range(num):
        tmp = m[i] / sum1
        nor.append(tmp)

    return nor


def calculateIntersect(string_1, string_2):
    """
    :param string_1: 字符串
    :param string_2: 字符串
    :return: 两字符串的交集
    """
    result = ''
    for char in string_1:
        if char in string_2 and char not in result:
            result += char
    return result


def calUnion(str1, str2):
    """
    :param string_1: 字符串
    :param string_2: 字符串
    :return: 两字符串的并集
    """
    result = str2
    for char in str1:
        if char not in str2:
            result += char
    return result


def Bel(m, fe):
    """calculate Bel function
    """
    length_fod = len(fe)
    res = np.zeros((1, length_fod))
    for i in range(length_fod):
        for j in range(length_fod):
            tmp = calculateIntersect(fe[i], fe[j])
            if tmp == fe[j]:  # B与A的交集如果等于B，则B属于A
                res[0][i] += m[0][j]  # bel function
    return res


def DST(mp1, mp2, P):
    """
    DS证据理论
    :param mp1: 证据源1，numpy数组，存储信度
    :param mp2: 证据源2，numpy数组，存储信度
    :param P: 辨识框架
    :return: 返回融合信度和冲突因子
    """
    length_fod = len(P)  # 幂集长度
    mp = np.zeros((1, length_fod), 'float64')  # 初始化最终结果mp
    k_matrix = np.zeros((length_fod, length_fod))  # 冲突因子乘子
    for k in range(length_fod):
        tmp = P[k]
        f_matrix = np.zeros((length_fod, length_fod))  # 融合乘子
        for i in range(length_fod):
            for j in range(length_fod):
                tmp_ij = calculateIntersect(P[i], P[j])  # 有无交集
                if not tmp_ij:  # 若空集
                    k_matrix[i][j] = 1
                if tmp_ij == tmp:  # 若交集等于P[k]
                    f_matrix[i][j] = 1
        mp[0][k] = sum(sum(np.dot(mp1.T, mp2) * f_matrix))
    k = sum(sum(np.dot(mp1.T, mp2) * k_matrix))
    mp = mp / (1 - k)
    return mp


def DSmC(m1, m2, P):
    """
       计算DSmC融合
    """
    length_fod = len(P)  # 幂集长度
    mf = np.zeros((1, length_fod), 'float64')  # 初始化最终结果mf
    for k in range(length_fod):
        tmp = P[k]
        f_matrix = np.zeros((length_fod, length_fod))  # 融合乘子
        for i in range(length_fod):
            for j in range(length_fod):
                tmp_ij = calculateIntersect(P[i], P[j])  # 有无交集
                if tmp_ij == tmp:  # 若交集等于P[k]
                    mf[0][k] += m1[0][i] * m2[0][j]
    return mf


def PCR5(m1, m2, P):
    """
    PCR5融合公式  该规则显然不满足结合律, 多个证据合成顺序影响最终结果
    第一步计算DSmC
    """
    mf1 = DSmC(m1, m2, P)
    """
    第二步计算信度再分配
    """
    length_fod = len(P)  # 幂集长度
    mf2 = np.zeros((1, length_fod), 'float64')
    for i in range(length_fod):
        s1 = 0
        s2 = 0
        for j in range(length_fod):
            tmp = calculateIntersect(P[i], P[j])
            if not tmp:
                if m1[0][i] + m2[0][j] != 0:
                    s1 += (m1[0][i] * m1[0][i] * m2[0][j]) / (m1[0][i] + m2[0][j])
                if m2[0][i] + m1[0][j] != 0:
                    s2 += (m2[0][i] * m2[0][i] * m1[0][j]) / (m2[0][i] + m1[0][j])

        mf2[0][i] = s1 + s2
    mf = mf1 + mf2
    return mf


def Murphy(BoE, P):
    """
    """
    num_SoE = BoE.shape[0]
    num_FE = BoE.shape[2]
    AE = np.zeros((1, num_FE))
    for j in range(num_FE):
        tmp = 0
        for i in range(num_SoE):
            tmp += BoE[i][0][j]
        AE[0][j] = tmp / num_SoE

    temp = AE
    for k in range(num_SoE - 1):
        temp = DST(temp, AE, P)
    return temp


def Murphy_weight(BoE, P, w):
    """
    """
    num_SoE = BoE.shape[0]
    num_FE = BoE.shape[2]
    AE = np.zeros((1, num_FE))
    for j in range(num_FE):
        tmp = 0
        for i in range(num_SoE):
            tmp += BoE[i][0][j] * w[i]
        AE[0][j] = tmp

    # print("AE=", AE)
    temp = AE
    for k in range(num_SoE - 1):
        temp = DST(temp, AE, P)
    return temp


def average_fusion(BoE):
    score = []
    num_FE = BoE.shape[2]
    num_SoE = BoE.shape[0]
    for i in range(num_FE):
        tmp = 0
        for j in range(num_SoE):
            tmp += BoE[j][0][i]
        score.append(tmp / num_FE)
    return score


def majority_voting(BoE):
    """
    只考虑贝叶斯BBA，如果不是，先做BetP
    如果存在两个类别的票数一致，那么无法决策，按照错误决策处理
    """
    num_FE = BoE.shape[2]
    num_SoE = BoE.shape[0]
    res = np.zeros((1, num_FE))
    for j in range(num_SoE):
        _pre = np.argmax(BoE[j], axis=1)
        res[0][_pre] += 1

    return res


def multiSourceFusion(boes, P):
    """
     采用DST实现多源信息融合 ,多个信息源一个一个的融合
    :param boes: 多个证据源存入一个np中
    """
    num_source = boes.shape[0]
    temp = boes[0]
    for i in range(num_source - 1):
        temp = DST(temp, boes[i + 1], P)
    return temp


def multiSourceFusion_PCR5(boes, P):
    num_source = boes.shape[0]
    temp = boes[0]
    for i in range(num_source - 1):
        temp = PCR5(temp, boes[i + 1], P)
    return temp


def multiSourceFusion_DSmC(boes, P):
    """
     采用DST实现多源信息融合 ,多个信息源一个一个的融合
    :param boes: 多个证据源存入一个np中
    """
    num_source = boes.shape[0]
    temp = boes[0]
    for i in range(num_source - 1):
        temp = DSmC(temp, boes[i + 1], P)
    return temp


def discountFusion(boes, P, weighted):
    """
         采用DST实现多源信息折扣融合 ,多个信息源一个一个的融合,
        :param boes: 多个证据源存入一个np中
    """
    # 原始BBA乘以权重因子
    num_source = boes.shape[0]
    # temp_boes = np.zeros((6, 1, 7))
    temp_boes = boes  # 这种直接赋值的方式在后续的运算中会直接改变boes中的值

    for j in range(num_source):
        temp_boes[j] = weighted[j] * boes[j]
        temp_boes[j][0][len(P) - 1] += (1 - weighted[j])

    res = temp_boes[0]
    for i in range(num_source - 1):
        # res = DST(res, temp_boes[i + 1], P)
        res = PCR5(res, temp_boes[i + 1], P)
    return res


def BetP(m, P):
    """
        Pignistic Probability Transform based on Smets's TBM
    """
    pt = []
    for i in range(len(P)):
        if len(P[i]) == 1:
            ppt = 0
            for j in range(len(P)):
                if P[i] in P[j]:  # 仅对单子焦元分配信度
                    ppt += m[0][j] / len(P[j])
            pt.append(ppt)
    return np.array(pt)
