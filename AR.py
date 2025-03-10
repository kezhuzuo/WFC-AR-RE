import fusionRules
import numpy as np

import similarityMeasure
import statistics


def BBA_approximation(BoE, P):
    tmp_BoE = BoE
    num = len(P)
    num_BBA = BoE.shape[0]
    FE_da = []
    for i in range(num_BBA):
        score = sorted(tmp_BoE[i][0], reverse=True)
        idx1 = tmp_BoE[i][0].tolist().index(score[0])
        idx2 = tmp_BoE[i][0].tolist().index(score[1])
        FE_da.append(P[idx1])
        FE_da.append(P[idx2])

    # 求解FE_DA集合中元素的并集
    FE_X = FE_da[0]
    for k in range(len(FE_da) - 1):
        FE_X = fusionRules.calUnion(FE_X, FE_da[k + 1])

    # 最终的焦元
    P1 = []
    for j in range(num):
        if fusionRules.calculateIntersect(FE_X, P[j]):
            P1.append(P[j])

    # 得到最终保留的焦元
    P1.append(FE_X)

    # 信度更新
    num_final_FE = len(P1)
    BoE_app = np.zeros((num_BBA, 1, num_final_FE))

    for m in range(num_final_FE - 1):
        id = P.index(P1[m])
        for n in range(num_BBA):
            BoE_app[n][0][m] = BoE[n][0][id]

    for p in range(num_BBA):
        BoE_app[p][0][num_final_FE - 1] = 1 - sum(BoE_app[p][0])

    return BoE_app, P1


def focal_element_evaluation(BoE, P):
    score = []
    num_FE = len(P)
    num_SoE = BoE.shape[0]
    for i in range(num_FE):
        tmp = 0
        for j in range(num_SoE):
            tmp += BoE[j][0][i] / len(P[i])
        score.append(tmp)
    return score


def normalizing_array(m):
    nm = np.zeros(m.shape)
    sum1 = 0
    for i in range(m.shape[1]):
        sum1 += m[0][i]
    for j in range(m.shape[1]):
        nm[0][j] = m[0][j] / sum1
    return nm


def mean_time(t):
    mean_ = statistics.mean(t)
    pst_ = statistics.pstdev(t)
    return mean_, pst_
