import math
from mpl_toolkits.mplot3d import Axes3D  # 画3d图案所必需的

import SOTA

import similarityMeasure
import numpy as np
import time
import fusionRules
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True, precision=3)


# The differences between BJS and BDM.

A = []
B = []
d1 = []
d2 = []

P = ['1', '123']
for i in range(21):
    a = i / 20

    if a == 0:
        a = a + 0.0001

    if a == 1:
        a = a - 0.0001

    m1 = np.array([[a, 1 - a]])
    m2 = np.array([[1 - a, a]])

    boe = np.zeros((2, 1, 2))
    boe[0] = m1
    boe[1] = m2

    A.append(a)
    B.append(a)

    row_d1 = []
    row_d2 = []
    for j in range(21):
        w1 = j / 20
        coe = [w1, 1 - w1]
        div1 = similarityMeasure.BJS(m1, m2)
        div2 = similarityMeasure.improved_divergence(m1, m2, P, coe)

        row_d1.append(div1)
        row_d2.append(div2)
        print('div1 =', div1)
        print('div2 =', div2)

    d1.append(row_d1)
    d2.append(row_d2)

# 转换为NumPy数组
A = np.array(A)
B = np.array(B)
x, y = np.meshgrid(A, B)
d1 = np.array(d1)
d2 = np.array(d2)

# 创建一个3D坐标轴
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 使用ax.plot_surface绘制3D曲面图
ax.plot_surface(x, y[::-1], d2, cmap='viridis')  # 反转 y 轴数据

# 设置坐标轴标签
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# 显示图形
plt.show()
