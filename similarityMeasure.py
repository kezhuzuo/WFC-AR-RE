import math
import numpy as np
import fusionRules
from fusionRules import calculateIntersect
from fusionRules import calUnion
import scipy.spatial


def BelPl(m, fe):
    """calculate Bel and Pl function
    """
    length_fod = len(fe)
    res = np.zeros((length_fod, 2), 'float64')  # 初始化BI

    for i in range(length_fod):
        for j in range(length_fod):
            tmp = calculateIntersect(fe[i], fe[j])
            if tmp:
                res[i][1] += m[0][j]  # pl function
                if tmp == fe[j]:  # B与A的交集如果等于B，则B属于A
                    res[i][0] += m[0][j]  # bel function
    return res


def BJS(m1, m2):
    """calculate the BJS divergence between two BBAs
    注意m1[i] + m2[i]不能为0，且log(x,2)中的x不能为0
    距离度量需要满足3个条件：
    （1）非负性
    （2）反身性
    （3）三角不等式
    """
    res = 0
    for i in range(m1.shape[1]):
        s1 = 0
        s2 = 0
        if m1[0][i] != 0:
            s1 = 0.5 * m1[0][i] * math.log(2 * m1[0][i] / (m1[0][i] + m2[0][i]), 2)
        if m2[0][i] != 0:
            s2 = 0.5 * m2[0][i] * math.log(2 * m2[0][i] / (m1[0][i] + m2[0][i]), 2)
        res += (s1 + s2)
    return res


def dengEntropy(m, P):
    """calculate the deng Entropy
    注意对于log函数 信度赋值不能为0
    """
    Ed = 0
    for i in range(len(P)):
        cardinality = len(P[i])
        if m[0][i] != 0:
            tmp = m[0][i] / (pow(2, cardinality) - 1)
            Ed += (m[0][i] * math.log(tmp, 2))
    return -Ed


def cross_similarity(m1, m2):
    ce1 = crossEntropy(m1, m1)
    ce2 = crossEntropy(m2, m2)
    ce3 = crossEntropy(m1, m2)
    ce4 = crossEntropy(m2, m1)
    res = (ce1 + ce2) / (ce3 + ce4)

    return res


def JousselmeDistance(m1, m2, P):
    """
    Jousselme距离相似度度量方法：A new distance between two bodies of evidence
    :param m1: 需要将m1看成是向量
    :param m2: 需要将m2看成是向量
    后面是两个一维数组的马氏距离计算方法
    :param P:
    """
    D = np.zeros((len(P), len(P)), 'float64')
    for i in range(len(P)):
        for j in range(len(P)):
            aib = calculateIntersect(P[i], P[j])
            aub = calUnion(P[i], P[j])
            D[i][j] = len(aib) / len(aub)

    # vec = m1 - m2
    # tmp = 0.5 * np.dot(np.dot(vec, D), vec.T)  # np.dot 是矩阵运算
    # res = pow(tmp, 0.5)
    res = scipy.spatial.distance.mahalanobis(m1.flatten(), m2.flatten(), D) * math.sqrt(0.5)  # 马氏距离的计算方法，和上面的计算结果一样
    return res


def crossEntropy(m1, m2):
    Ec = 0
    for i in range(m1.shape[1]):
        if m2[0][i] != 0:
            Ec += (m1[0][i] * math.log(m2[0][i], 2) * (-1))
    return Ec


def negationEvidence(mp1, P):
    """计算证据源的negation
    """
    res = np.zeros((mp1.shape))
    if len(P) == 1:
        print("negation is 0")
    else:
        for i in range(len(P)):
            res[0][i] = (1 - mp1[0][i]) / (len(P) - 1)

    return res


def RBJS(m1, m2, P1, P2):
    """calculate the RBJS divergence between two BBAs
    注意m1[i] + m2[i]不能为0，且log(x,2)中的x不能为0
    距离度量需要满足3个条件：
    （1）非负性
    （2）反身性
    （3）三角不等式
    """
    res = 0
    for i in range(m1.shape[1]):
        for j in range(m2.shape[1]):
            s1 = 0
            s2 = 0
            tmp = calculateIntersect(P1[i], P2[j])
            card1 = len(tmp) / len(P2[j])
            card2 = len(tmp) / len(P1[i])
            if m1[0][i] != 0 and card1 != 0:
                s1 = m1[0][i] * math.log(2 * card1 * m1[0][i] / (m1[0][i] + m2[0][j]), 2)
            if m2[0][j] != 0 and card2 != 0:
                s2 = m2[0][j] * math.log(2 * card2 * m2[0][j] / (m1[0][i] + m2[0][j]), 2)
            res += ((s1 + s2) * 0.5)  # xiao的论文中公式没有✖0.5，但是需要乘以0.5，结果才与她的算例一样
    return res


def RB(m1, m2, P1, P2):
    B1 = RBJS(m1, m1, P1, P1)
    B2 = RBJS(m2, m2, P2, P2)
    B3 = RBJS(m1, m2, P1, P2)
    res = math.sqrt(abs(B1 + B2 - 2 * B3) / 2)
    return res


def focal_element_distance_union(m, P):
    num = len(P)
    res = np.zeros((num, num))
    for i in range(num):
        for j in range(num):
            fe_union = fusionRules.calUnion(P[i], P[j])
            res[i][j] = (m[0][i] + m[0][j]) * len(fe_union) - m[0][i] * len(P[i]) - m[0][j] * len(P[j])

    return res


def focal_element_distance_intersection(m, P):
    num = len(P)
    res = np.zeros((num, num))
    for i in range(num):
        for j in range(num):
            fe_intersection = fusionRules.calculateIntersect(P[i], P[j])
            res[i][j] = m[0][i] * len(P[i]) + m[0][j] * len(P[j]) - (m[0][i] + m[0][j]) * len(fe_intersection)

    return res


def gao_cross_entropy(m1, m2, P):
    num = len(P)
    CE = 0
    for i in range(num):
        crd = len(P[i])
        if m2[0][i] != 0:
            tmp = m2[0][i] / (pow(2, crd) - 1)
            CE += m1[0][i] * math.log(tmp, 2)
    return -CE


def gao_similarity(m1, m2, P):
    ce1 = gao_cross_entropy(m1, m1, P)
    ce2 = gao_cross_entropy(m2, m2, P)
    ce3 = gao_cross_entropy(m1, m2, P)
    ce4 = gao_cross_entropy(m2, m1, P)
    res = (ce1 + ce2) / (ce3 + ce4)

    return res


# BDM method in IEEE TSMC
def improved_divergence(m1, m2, P, w):
    BF1 = BelPl(m1, P)
    BF2 = BelPl(m2, P)
    num_FE = len(P)
    res = 0

    for i in range(num_FE):
        # if len(P[i]) == 1:
        s1 = 0
        s2 = 0
        u1 = 1 - (BF1[i][1] - BF1[i][0])
        u2 = 1 - (BF2[i][1] - BF2[i][0])
        u = w[0] * u1 + w[1] * u2
        # u = (u1 + u2) / 2
        if m1[0][i] / (w[0] * m1[0][i] + w[1] * m2[0][i]) > 0:
            s1 = m1[0][i] * math.log(m1[0][i] / (w[0] * m1[0][i] + w[1] * m2[0][i]), 2)
        if m2[0][i] / (w[0] * m1[0][i] + w[1] * m2[0][i]) > 0:
            s2 = m2[0][i] * math.log(m2[0][i] / (w[0] * m1[0][i] + w[1] * m2[0][i]), 2)
        res += ((w[0] * s1 + w[1] * s2) * u)

    return res


def GEJS(BoE, P, w):
    num_SoE = BoE.shape[0]
    num_FE = len(P)
    res = 0
    for j in range(num_SoE):
        s = 0
        for i in range(num_FE):
            u = 1 / len(P[i])
            tmp = 0
            for k in range(num_SoE):
                tmp += w[k] * BoE[k][0][i]

            if BoE[j][0][i] != 0 and tmp != 0:
                s += (u * BoE[j][0][i] * math.log(BoE[j][0][i] / tmp, 2))
        res += (w[j] * s)

    return res


def fusion_by_GEJS(BoE, P, w):
    num_SoE = BoE.shape[0]
    num_FE = len(P)

    # 计算总的差异性Div
    Div = GEJS(BoE, P, w)

    # 计算去掉第i个证据体的Div_i
    res = []
    wf = []
    for i in range(num_SoE):

        # 去掉第i个证据体和相应的证据体可靠度权重
        tmp = np.zeros((num_SoE - 1, 1, num_FE))
        idx = list(range(num_SoE))
        wc = list(w)
        del idx[i]
        del wc[i]

        # 证据可靠度权重新归一化
        sum_w = sum(wc)
        wn = []
        for p in range(num_SoE - 1):
            wn.append(wc[p] / sum_w)

        # 计算去掉第i个证据体的Div_i
        for j in range(num_SoE - 1):
            tmp[j] = BoE[idx[j]]

        Div_i = GEJS(tmp, P, wn)
        coe = Div_i / Div
        res.append(coe)

    # 计算每个证据体的最终权重
    wf = fusionRules.nor_list(res)
    com = fusionRules.Murphy_weight(BoE, P, wf)

    return com
