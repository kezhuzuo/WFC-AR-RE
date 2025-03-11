import fusionRules
import numpy as np


def BBA_approximation(BoE, P):
    """
    This function implements our approximation algorithm.
    """

    tmp_BoE = BoE
    num = len(P)
    num_BBA = BoE.shape[0]
    FE_da = []
    for i in range(num_BBA):
        score = sorted(tmp_BoE[i][0], reverse=True)
        idx1 = tmp_BoE[i][0].tolist().index(score[0])
        idx2 = tmp_BoE[i][0].tolist().index(score[1])

        # The two elements with the highest belief values in each BBA
        FE_da.append(P[idx1])
        FE_da.append(P[idx2])

    # Compute the union of elements in the FE_DA set
    FE_X = FE_da[0]
    for k in range(len(FE_da) - 1):
        FE_X = fusionRules.calUnion(FE_X, FE_da[k + 1])

    # Retained focal elements
    P1 = []
    for j in range(num):
        if fusionRules.calculateIntersect(FE_X, P[j]):
            P1.append(P[j])

    # Obtain the final focal elements in the approximated BBA
    P1.append(FE_X)

    # Belief update
    num_final_FE = len(P1)
    BoE_app = np.zeros((num_BBA, 1, num_final_FE))

    for m in range(num_final_FE - 1):
        id = P.index(P1[m])
        for n in range(num_BBA):
            BoE_app[n][0][m] = BoE[n][0][id]

    for p in range(num_BBA):
        BoE_app[p][0][num_final_FE - 1] = 1 - sum(BoE_app[p][0])

    return BoE_app, P1
