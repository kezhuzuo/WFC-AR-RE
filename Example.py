import math
import SOTA
import similarityMeasure
import numpy as np
import time
import fusionRules
import AR

np.set_printoptions(suppress=True, precision=3)

# Examples in  section IV

fod = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
m1 = np.array([[0., 0., 0., 0., 0.01, 0.02, 0.01, 0.02, 0, 0.41, 0.37, 0.16]])
m2 = np.array([[0., 0., 0.02, 0.0, 0.02, 0.01, 0.01, 0.01, 0.01, 0.57, 0.13, 0.22]])
m3 = np.array([[0.0, 0., 0.03, 0., 0.02, 0., 0.04, 0.02, 0.02, 0.06, 0.62, 0.19]])
sr = [0.963, 0.948, 0.716]

BoE = np.zeros((3, 1, 12))  # BoE denote the array composed of K original BBAs
BoE[0] = m1
BoE[1] = m2
BoE[2] = m3

t_dst = []
t_mur = []
t_xiao = []
t_tang = []
t_WFC_AR_RE = []

# for i in range(50):
# -------------DS-------------------
start_dst = time.clock()
res_dst = fusionRules.multiSourceFusion(BoE, fod)
end_dst = time.clock()
t_dst.append((end_dst - start_dst) * 1000)

# -------------Murphy-------------------
start_mur = time.clock()
mur = fusionRules.Murphy(BoE, fod)
end_mur = time.clock()
t_mur.append((end_mur - start_mur) * 1000)

# -------------xiao_BJS-------------------
start_xiao = time.clock()
res_BJS = SOTA.xiao(BoE, fod, 3)
end_xiao = time.clock()
t_xiao.append((end_xiao - start_xiao) * 1000)

# -------------Tang-------------------
start_tang = time.clock()
tang = SOTA.measure_uncertainty_negation(BoE, fod, 3)
end_tang = time.clock()
t_tang.append((end_tang - start_tang) * 1000)

# -----------------WFC_AR_RE-------------------------------
start_WFC_AR_RE = time.clock()
BBA, P1 = AR.BBA_approximation(BoE, fod)  # Perform approximation for the original BBA
srn = fusionRules.nor_list(sr)
WFC_AR_RE = SOTA.zuo(BBA, P1, srn, 3)
end_WFC_AR_RE = time.clock()
t_WFC_AR_RE.append((end_WFC_AR_RE - start_WFC_AR_RE) * 1000)

print('dst = ', res_dst)
print('mur = ', mur)
print('BJS =', res_BJS)
print("tang =", tang)
print('WFC_AR_RE =', WFC_AR_RE)

# # ---------------------------时间对比-----------------------------------
# mean_dst, pst_dst = AR.mean_time(t_dst)
# mean_mur, pst_mur = AR.mean_time(t_mur)
# mean_xiao, pst_xiao = AR.mean_time(t_xiao)
# mean_RB, pst_RB = AR.mean_time(t_RB)
# mean_tang, pst_tang = AR.mean_time(t_tang)
# mean_WFC_AR_RE, pst_WFC_AR_RE = AR.mean_time(t_WFC_AR_RE)
#
# print('time_dst =', mean_dst, pst_dst)
# print('time_mur =', mean_mur, pst_mur)
# print('time_xiao =', mean_xiao, pst_xiao)
# print('time_RB =', mean_RB, pst_RB)
# print('time_tang =', mean_tang, pst_tang)
# print('time_WFC_AR_RE =', mean_WFC_AR_RE, pst_WFC_AR_RE)
