import math
import numpy as np
import similarityMeasure
import fusionRules

from fusionRules import calculateIntersect
from fusionRules import calUnion
import scipy.spatial


def xiao(BoE, P, num_source):
    BJS = []
    num_FE = len(P)
    for i in range(num_source):
        tmp_BJS = 0
        for j in range(num_source):
            if j != i:
                tmp_BJS += similarityMeasure.BJS(BoE[i], BoE[j])
        BJS.append(((num_source - 1) / tmp_BJS))
        # BJS.append(1 - (tmp_BJS / (num_source - 1)))

    Crd = []
    sum1 = sum(BJS)
    for k in range(num_source):
        Crd.append(BJS[k] / sum1)

    Ed = []
    for i in range(num_source):
        tmp_Ed = similarityMeasure.dengEntropy(BoE[i], P)
        Ed.append(math.exp(tmp_Ed))

    IV = []
    sum2 = sum(Ed)
    for m in range(num_source):
        IV.append(Ed[m] / sum2)

    ACrd = []
    for n in range(num_source):
        ACrd.append(Crd[n] * IV[n])

    w = []
    sum3 = sum(ACrd)
    for i in range(num_source):
        w.append(ACrd[i] / sum3)

    AE = np.zeros((1, num_FE))
    for j in range(num_FE):
        for i in range(num_source):
            AE[0][j] += BoE[i][0][j] * w[i]

    temp = AE
    for k in range(num_source - 1):
        temp = fusionRules.DST(temp, AE, P)
    return temp


def measure_uncertainty_negation(BoE, P, num_source):
    negation_BoE = np.zeros((BoE.shape))
    num_FE = BoE.shape[2]
    for i in range(num_source):
        for j in range(num_FE):
            negation_BoE[i] = similarityMeasure.negationEvidence(BoE[i], P)

    Ed = []
    for i in range(num_source):
        tmp_Ed = similarityMeasure.dengEntropy(negation_BoE[i], P)
        Ed.append(math.exp(tmp_Ed))

    IV = []
    sum2 = sum(Ed)
    for m in range(num_source):
        IV.append(Ed[m] / sum2)

    AE = np.zeros((1, num_FE))
    for j in range(num_FE):
        for i in range(num_source):
            AE[0][j] += BoE[i][0][j] * IV[i]

    temp = AE
    for k in range(num_source - 1):
        temp = fusionRules.DST(temp, AE, P)
    return temp


def cross_entropy_similarity(BoE, P, num_source):
    CE = []
    num_FE = len(P)
    for i in range(num_source):
        tmp_CE = 0
        for j in range(num_source):
            tmp_CE += similarityMeasure.gao_cross_entropy(BoE[i], BoE[j], P)
        CE.append(tmp_CE)

    Crd = []
    sum1 = sum(CE)
    for k in range(num_source):
        Crd.append(CE[k] / sum1)

    AE = np.zeros((1, num_FE))
    for j in range(num_FE):
        for i in range(num_source):
            AE[0][j] += BoE[i][0][j] * Crd[i]

    temp = AE
    for k in range(num_source - 1):
        temp = fusionRules.DST(temp, AE, P)
    return temp


def enhanced_voting(BoE, acc):
    """
    只考虑贝叶斯BBA，如果不是，先做BetP
    如果存在两个类别的票数一致，那么无法决策，按照错误决策处理
    """
    sum1 = sum(acc)
    w = []
    num_SoE = BoE.shape[0]
    for i in range(num_SoE):
        w.append(acc[i] / sum1)

    num_FE = BoE.shape[2]
    res = np.zeros((1, num_FE))
    for j in range(num_SoE):
        _pre = np.argmax(BoE[j], axis=1)
        res[0][_pre] += w[j]

    return res


def zhao_distance(BoE, P, num_source):
    RB = []
    num_FE = len(P)
    for i in range(num_source):
        tmp_RB = 0
        for j in range(num_source):
            if j != i:
                tmp_RB += similarityMeasure.JousselmeDistance(BoE[i], BoE[j], P)
        RB.append((num_source - 1) / tmp_RB)

    w = fusionRules.nor_list(RB)

    AE = np.zeros((1, num_FE))
    for j in range(num_FE):
        for i in range(num_source):
            AE[0][j] += BoE[i][0][j] * w[i]

    temp = AE
    for k in range(num_source - 1):
        temp = fusionRules.DST(temp, AE, P)

    return temp


def zuo(BoE, P, sr, num_source):
    w = []
    # 计算BBA的支持度sup
    for p in range(num_source):
        tmp_jou = 0
        for q in range(num_source):
            if q != p:
                tmp_jou += (1 - similarityMeasure.JousselmeDistance(BoE[p], BoE[q], P))
        w.append(tmp_jou / (num_source - 1))

    # 根据所提的BDM方法计算每个BBA的差异性
    JS = []
    for i in range(num_source):
        wi = w[i]
        tmp_BDM = 0
        for j in range(num_source):
            if j != i:
                wj = w[j]
                w1 = wi / (wi + wj)
                coe = [w1, 1 - w1]
                tmp_BDM += similarityMeasure.BDM(BoE[i], BoE[j], P, coe)

        JS.append(((num_source - 1) / tmp_BDM))

    # 计算每个BBA的权重
    f = []
    for k in range(num_source):
        f.append((JS[k] * sr[k]))

    wc = fusionRules.nor_list(f)
    # print('wc =', wc)

    com = fusionRules.Murphy_weight(BoE, P, wc)

    return com
