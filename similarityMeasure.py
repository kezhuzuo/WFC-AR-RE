import math
import numpy as np
from fusionRules import calculateIntersect
from fusionRules import calUnion
import scipy.spatial


def BelPl(m, fe):
    """Calculate Belief (Bel) and Plausibility (Pl) functions."""
    length_fod = len(fe)
    res = np.zeros((length_fod, 2), 'float64')  # Initialize BI

    for i in range(length_fod):
        for j in range(length_fod):
            tmp = calculateIntersect(fe[i], fe[j])
            if tmp:
                res[i][1] += m[0][j]  # Pl function
                if tmp == fe[j]:  # If the intersection of B and A equals B, then B is a subset of A
                    res[i][0] += m[0][j]  # Bel function
    return res


def BJS(m1, m2):
    """Calculate the BJS divergence between two BBAs.
    Note: m1[i] + m2[i] cannot be 0, and log(x,2) cannot take x=0.
    A valid distance measure must satisfy three conditions:
    (1) Non-negativity
    (2) Reflexivity
    (3) Triangle inequality
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
    """Calculate Deng entropy.
    Note: The belief assignment cannot be 0 for the log function.
    """
    Ed = 0
    for i in range(len(P)):
        cardinality = len(P[i])
        if m[0][i] != 0:
            tmp = m[0][i] / (pow(2, cardinality) - 1)
            Ed += (m[0][i] * math.log(tmp, 2))
    return -Ed


def JousselmeDistance(m1, m2, P):
    """Calculate the Jousselme distance similarity measure.
    "A new distance between two bodies of evidence."
    :param m1: Treat m1 as a vector
    :param m2: Treat m2 as a vector
    :param P: The set of focal elements
    """
    D = np.zeros((len(P), len(P)), 'float64')
    for i in range(len(P)):
        for j in range(len(P)):
            aib = calculateIntersect(P[i], P[j])
            aub = calUnion(P[i], P[j])
            D[i][j] = len(aib) / len(aub)

    res = scipy.spatial.distance.mahalanobis(m1.flatten(), m2.flatten(), D) * math.sqrt(0.5)
    return res


def crossEntropy(m1, m2):
    """Calculate cross-entropy."""
    Ec = 0
    for i in range(m1.shape[1]):
        if m2[0][i] != 0:
            Ec += (m1[0][i] * math.log(m2[0][i], 2) * (-1))
    return Ec


def negationEvidence(mp1, P):
    """Calculate the negation of a body of evidence."""
    res = np.zeros((mp1.shape))
    if len(P) == 1:
        print("Negation is 0")
    else:
        for i in range(len(P)):
            res[0][i] = (1 - mp1[0][i]) / (len(P) - 1)
    return res


def gao_cross_entropy(m1, m2, P):
    """Calculate Gao's cross-entropy measure."""
    num = len(P)
    CE = 0
    for i in range(num):
        crd = len(P[i])
        if m2[0][i] != 0:
            tmp = m2[0][i] / (pow(2, crd) - 1)
            CE += m1[0][i] * math.log(tmp, 2)
    return -CE


def gao_similarity(m1, m2, P):
    """Calculate Gao's similarity measure."""
    ce1 = gao_cross_entropy(m1, m1, P)
    ce2 = gao_cross_entropy(m2, m2, P)
    ce3 = gao_cross_entropy(m1, m2, P)
    ce4 = gao_cross_entropy(m2, m1, P)
    res = (ce1 + ce2) / (ce3 + ce4)
    return res


def BDM(m1, m2, P, w):
    """BDM method in IEEE TSMC."""
    BF1 = BelPl(m1, P)
    BF2 = BelPl(m2, P)
    num_FE = len(P)
    res = 0

    for i in range(num_FE):
        s1 = 0
        s2 = 0
        u1 = 1 - (BF1[i][1] - BF1[i][0])
        u2 = 1 - (BF2[i][1] - BF2[i][0])
        u = w[0] * u1 + w[1] * u2

        if m1[0][i] / (w[0] * m1[0][i] + w[1] * m2[0][i]) > 0:
            s1 = m1[0][i] * math.log(m1[0][i] / (w[0] * m1[0][i] + w[1] * m2[0][i]), 2)
        if m2[0][i] / (w[0] * m1[0][i] + w[1] * m2[0][i]) > 0:
            s2 = m2[0][i] * math.log(m2[0][i] / (w[0] * m1[0][i] + w[1] * m2[0][i]), 2)
        res += ((w[0] * s1 + w[1] * s2) * u)
    return res
