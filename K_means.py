from numpy import *
import random


def initK():
    k = input("请输入聚类簇数：")
    return k


def init_C(k):
    """初始化C集合中到底分几个簇"""
    C = {"type1": []}
    for t in range(k):
        C["type%d" % (t + 1)] = []

    return C


def loadData():
    """将文本文件dataSet写入矩阵"""
    fr = open('dataSet')
    arrys = fr.readlines()
    number_lines = len(arrys)
    return_mat = zeros((number_lines, 2))  # 创建矩阵时一定是两个括号
    index = 0
    for line in arrys:
        line = line.strip()  # 去掉空格
        list_from_line = line.split(' ')
        return_mat[index, :] = list_from_line[1:3]
        index += 1
    return return_mat


# 全局的变量
k = (int)(initK())  # 簇数
init_mat = loadData()  # 放样本的矩阵
m = init_mat.shape[0]  # 矩阵的行
n = init_mat.shape[1]  # 矩阵的列


def k_means():
    """第一次时随机选取u向量和计算距离等"""
    u_mat = zeros((k, n))  # 初始化u向量组成的矩阵，用zeros创建
    a = random.randint(0, m - 1)  # 随机选取样本
    u_mat[0, :] = init_mat[a, :]
    for i in range(k - 1):
        b = random.randint(0, m - 1)
        if b != a:
            u_mat[i + 1, :] = init_mat[b, :]
    print("初始化的u矩阵为：")
    print(u_mat)
    C = init_C(k)  # 初始化C集合 C{'type':['x1','x2','x3'...],'type2'}
    dist_mat = zeros((k, 1))  # 存放样本与u向量距离的矩阵
    for j in range(m):
        for t in range(k):
            dist_mat[t, :] = math.sqrt((init_mat[j, 0] - u_mat[t, 0]) ** 2 + (init_mat[j, 1] - u_mat[t, 1]) ** 2)
        dist_index = dist_mat.argmin()  # 得到dist_mat矩阵中最短距离所在的索引
        C["type%d" % (dist_index + 1)].append("x%d" % (j + 1))  # 索引与我们习惯的样本1，样本2直接差了1

    while True:
        update_C, update_u_mat = updateMat(C, u_mat)
        if update_C == C:
            break
        else:
            C = update_C
            u_mat = update_u_mat
    print(C)


def updateMat(C, u_mat):
    """更新u向量组成的矩阵与C的分类集合"""
    for i in range(k):
        temp_mat = zeros((len(C["type%d" % (i + 1)]), n))
        for j in range(len(C["type%d" % (i + 1)])):
            """想要取出C {'type1': ['x5', 'x6', 'x7', 'x8'], 'type2': ['x1', 'x2', 'x3', 'x4']}
            中的1，有点蠢哈哈
            """
            temp_index = (int)((C["type%d" % (i + 1)][j])[1]) - 1
            temp_mat[j, :] = init_mat[temp_index, :]
        u = updateVector(temp_mat)  # 得到了当前Ci的矩阵--》去计算u
        if u_mat[i, :].all == u.all:  # 在判断数组或是向量相不相等时用a.all 这个是只读的，意思就是不能赋值
            break
        else:
            u_mat[i, :] = u
    print("更新之后的u矩阵为：")
    print(u_mat)
    C = init_C(k)
    dist_mat = zeros((k, 1))
    for j in range(m):
        for t in range(k):
            dist_mat[t, :] = math.sqrt((init_mat[j, 0] - u_mat[t, 0]) ** 2 + (init_mat[j, 1] - u_mat[t, 1]) ** 2)
        dist_index = dist_mat.argmin()
        C["type%d" % (dist_index + 1)].append("x%d" % (j + 1))

    return C, u_mat


def updateVector(mat):
    """更新u向量"""
    numbers_stamp = mat.shape[0]  # 样本个数，即矩阵的行数
    # 计算新u的公式为所有样本对应的x坐标相加再除以该簇Ci中所含样本的个数
    return_u = mat.sum(axis=0)  # 矩阵列之和
    if numbers_stamp == 0:
        return_u = zeros((1, n))
    else:
        return_u = return_u / numbers_stamp
    return return_u  # 返回该u向量
