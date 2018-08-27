import K_means
import matplotlib.pyplot as plt
from numpy import *


def loadDatas():
    """将文本文件dataSet写入矩阵"""
    fr = open('dataSet')
    arrys = fr.readlines()
    number_lines = len(arrys)
    return_mat = zeros((number_lines, 3))  # 创建矩阵时一定是两个括号
    label_mat = zeros((1, number_lines))
    index = 0
    for line in arrys:
        line = line.strip()  # 去掉空格
        list_from_line = line.split(' ')
        return_mat[index, :] = list_from_line[0:3]
        label_mat[0, index] = list_from_line[0]
        index += 1
    return return_mat, label_mat


mat, labels = loadDatas()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(mat[:, 0], mat[:, 1])
plt.show()
K_means.k_means()
