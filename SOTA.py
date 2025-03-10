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


def xiao_RB(BoE, P, num_source):
    RB = []
    num_FE = len(P)
    for i in range(num_source):
        tmp_RB = 0
        for j in range(num_source):
            if j != i:
                tmp_RB += similarityMeasure.RB(BoE[i], BoE[j], P, P)
        RB.append(((num_source - 1) / tmp_RB))

    w = fusionRules.nor_list(RB)

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

def generate_BBA(prob, P):
    """
    follow文献'A target intention recognition method based on information classification
                processing and information fusion'_jiang_wen
    """
    num = len(P)
    res = np.zeros((1, 2 * num))
    NE = similarityMeasure.negationEvidence(prob, P)
    for i in range(2 * num):
        if i < num:
            res[0][i] = prob[0][i] / 2
        else:
            res[0][i] = NE[0][i - num] / 2

    return res


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
                tmp_BDM += similarityMeasure.improved_divergence(BoE[i], BoE[j], P, coe)

        JS.append(((num_source - 1) / tmp_BDM))

    # 计算每个BBA的权重
    f = []
    for k in range(num_source):
        f.append((JS[k] * sr[k]))

    wc = fusionRules.nor_list(f)
    # print('wc =', wc)

    com = fusionRules.Murphy_weight(BoE, P, wc)

    return com


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
