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
    :param string_1: string
    :param string_2: string
    :return: Intersection of two strings
    """
    result = ''
    for char in string_1:
        if char in string_2 and char not in result:
            result += char
    return result


def calUnion(str1, str2):
    """
    :param str1: string
    :param str2: string
    :return: Union of two strings
    """
    result = str2
    for char in str1:
        if char not in str2:
            result += char
    return result


def Bel(m, fe):
    """Calculate Bel function"""
    length_fod = len(fe)
    res = np.zeros((1, length_fod))
    for i in range(length_fod):
        for j in range(length_fod):
            tmp = calculateIntersect(fe[i], fe[j])
            if tmp == fe[j]:  # If the intersection of B and A equals B, then B belongs to A
                res[0][i] += m[0][j]  # bel function
    return res


def DST(mp1, mp2, P):
    """
    Dempster-Shafer evidence theory
    :param mp1: Evidence source 1, numpy array storing belief values
    :param mp2: Evidence source 2, numpy array storing belief values
    :param P: Frame of discernment
    :return: Returns fused belief values and conflict factor
    """
    length_fod = len(P)  # Power set length
    mp = np.zeros((1, length_fod), 'float64')  # Initialize final result mp
    k_matrix = np.zeros((length_fod, length_fod))  # Conflict factor multiplier
    for k in range(length_fod):
        tmp = P[k]
        f_matrix = np.zeros((length_fod, length_fod))  # Fusion multiplier
        for i in range(length_fod):
            for j in range(length_fod):
                tmp_ij = calculateIntersect(P[i], P[j])  # Check for intersection
                if not tmp_ij:  # If empty set
                    k_matrix[i][j] = 1
                if tmp_ij == tmp:  # If intersection equals P[k]
                    f_matrix[i][j] = 1
        mp[0][k] = sum(sum(np.dot(mp1.T, mp2) * f_matrix))
    k = sum(sum(np.dot(mp1.T, mp2) * k_matrix))
    mp = mp / (1 - k)
    return mp


def DSmC(m1, m2, P):
    """
       Compute DSmC fusion
    """
    length_fod = len(P)  # Power set length
    mf = np.zeros((1, length_fod), 'float64')  # Initialize final result mf
    for k in range(length_fod):
        tmp = P[k]
        f_matrix = np.zeros((length_fod, length_fod))  # Fusion multiplier
        for i in range(length_fod):
            for j in range(length_fod):
                tmp_ij = calculateIntersect(P[i], P[j])  # Check for intersection
                if tmp_ij == tmp:  # If intersection equals P[k]
                    mf[0][k] += m1[0][i] * m2[0][j]
    return mf


def PCR6(m1, m2, P):
    """
    PCR6 fusion formula. This rule does not satisfy the associative law, meaning that the order of combining multiple pieces of evidence affects the final result.
    Step 1: Compute DSmC
    """
    mf1 = DSmC(m1, m2, P)
    """
    Step 2: Reallocate belief
    """
    length_fod = len(P)  # Power set length
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

    temp = AE
    for k in range(num_SoE - 1):
        temp = DST(temp, AE, P)
    return temp


def multiSourceFusion(boes, P):
    """
    Multi-source information fusion using DST, fusing multiple sources one by one
    :param boes: Multiple evidence sources stored in a numpy array
    """
    num_source = boes.shape[0]
    temp = boes[0]
    for i in range(num_source - 1):
        temp = DST(temp, boes[i + 1], P)
    return temp


def BetP(m, P):
    """
        Pignistic Probability Transform based on Smets's TBM
    """
    pt = []
    for i in range(len(P)):
        if len(P[i]) == 1:
            ppt = 0
            for j in range(len(P)):
                if P[i] in P[j]:  # Assign belief only to single focal elements
                    ppt += m[0][j] / len(P[j])
            pt.append(ppt)
    return np.array(pt)
