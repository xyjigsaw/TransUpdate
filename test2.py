# Name: test2
# Author: Reacubeth
# Time: 2020/3/27 17:31
# Mail: noverfitting@gmail.com
# Site: www.omegaxyz.com
# *_*coding:utf-8 *_*

# -*- coding: utf-8 -*-


from __future__ import division
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import coo_matrix
import datetime


# #准备函数#######################################################################

# 图形初始化
def process_params(G, center, dim):
    if type(G) != nx.Graph:
        empty_graph = nx.Graph()
        empty_graph.add_nodes_from(G)
        G = empty_graph

    if center is None:
        center = np.zeros(dim)
    else:
        center = np.asarray(center)

    if len(center) != dim:
        print("length of center coordinates must match dimension of layout")

    return G, center


# 坐标归一化
def rescale_layout(pos, scale=1):
    lim = 0
    for i in range(pos.shape[1]):
        pos[:, i] -= pos[:, i].mean()
        lim = max(abs(pos[:, i]).max(), lim)

    if lim > 0:
        for i in range(pos.shape[1]):
            pos[:, i] *= scale / lim
    return pos


# 稀疏矩阵形式的矩阵A
def to_scipy_sparse_matrix(G, nodelist=None, dtype=None,
                           weight='weight', format='csr'):
    if nodelist is None:
        nodelist = list(G)

    nlen = len(nodelist)

    if nlen == 0:
        print("Graph has no nodes or edges")

    if len(nodelist) != len(set(nodelist)):
        print("Ambiguous ordering: `nodelist` contained duplicates.")

    index = dict(zip(nodelist, range(nlen)))
    coefficients = zip(*((index[u], index[v], d.get(weight, 1))  # d.get(key,elsePrint),d是一个字典，通过键key获得字典的对应值，否则，令值为1
                         for u, v, d in G.edges(nodelist, data=True)  # u,v,d = 3773501 3870718 {}
                         if u in index and v in index))

    try:
        row, col, data = coefficients
    except ValueError:
        row, col, data = [], [], []

    # 对称矩阵
    d = data + data
    r = row + col
    c = col + row
    if [len(r), len(c)] != [len(d), len(d)]:
        print('坐标长度或坐标与数据的长度不相等')

    m = sparse.coo_matrix((d, (r, c)), shape=(nlen, nlen), dtype=dtype)
    # m.toarray()

    try:
        return m.asformat(format)
    except (AttributeError, ValueError):
        print("Unknown sparse matrix format: %s" % format)


# 数组形式的矩阵A
def to_numpy_array(G, nodelist=None, dtype=None, order=None,
                   multigraph_weight=sum, weight='weight', nonedge=0.0):
    if nodelist is None:
        nodelist = list(G)

    nodeset = set(nodelist)

    if len(nodelist) != len(nodeset):
        print('"Ambiguous ordering: `nodelist` contained duplicates."')

    nlen = len(nodelist)  # 点的个数
    index = dict(zip(nodelist, range(nlen)))
    A = np.full((nlen, nlen), np.nan, order=order)  # np.full((row,col)),value)建立row行col列值为value的数组

    for u, nbrdict in G.adjacency():
        print(u, nbrdict)
        for v, d in nbrdict.items():
            print(v, d)
            try:
                A[index[u], index[v]] = d.get(weight, 1)
            except KeyError:
                pass

    A[np.isnan(A)] = nonedge  # 没有边的位置用0填充
    A = np.asarray(A, dtype=dtype)  # A是对称矩阵，如果x_ij不为0，表示第i（代表的点）行与第j（代表的点）列之间有边
    return A


# 点个数较多时各点坐标的计算：稀疏矩阵
def sparse_fruchterman_reingold(A, k=None, pos=None, fixed=None, iterations=50, threshold=1e-4, dim=2):
    np.random.seed(1)

    try:
        nnodes, _ = A.shape
    except AttributeError:
        print('fruchterman_reingold() takes an adjacency matrix as input')

    try:
        A = A.tolil()

    except:
        A = (coo_matrix(A)).tolil()

    if pos is None:
        pos = np.asarray(np.random.rand(nnodes, dim), dtype=A.dtype)
    else:
        pos = pos.astype(A.dtype)

    if fixed is None:
        fixed = []

    if k is None:
        k = np.sqrt(1.0 / nnodes)
    t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1])) * 0.1
    dt = t / float(iterations + 1)

    displacement = np.zeros((dim, nnodes))
    for iteration in range(iterations):
        displacement *= 0  # 数组的形状不变，每个值都为0
        for i in range(A.shape[0]):  # 行数
            if i in fixed:  # fixed是数组，保存固定不变的坐标的索引，若i在索引中，则跳过以下步骤
                continue
            delta = (pos[i] - pos).T  # pos[0]是第零行，pos[0]是一行，用于减去pos的每一行
            distance = np.sqrt((delta ** 2).sum(axis=0))  # delta**2是数组中的每个值都平方;axis=0每一列相加
            distance = np.where(distance < 0.01, 0.01, distance)
            Ai = np.asarray(A.getrowview(i).toarray())
            displacement[:, i] += \
                (delta * (k * k / distance ** 2 - Ai * distance / k)).sum(axis=1)
        # update positions 更新位置
        length = np.sqrt((displacement ** 2).sum(axis=0))
        length = np.where(length < 0.01, 0.1, length)
        delta_pos = (displacement * t / length).T
        pos += delta_pos
        # cool temperature 冷却温度
        t -= dt
        err = np.linalg.norm(delta_pos) / nnodes
        if err < threshold:
            break
    return pos


# 点个数较少时各点坐标的计算：数组
def fruchterman_reingold(A, k=None, pos=None, fixed=None, iterations=50,
                         threshold=1e-4, dim=2):
    np.random.seed(2)
    nnodes, _ = A.shape  # nnodes是矩阵A的行数

    if pos is None:
        pos = np.asarray(np.random.rand(nnodes, dim))

    if k is None:
        k = np.sqrt(1.0 / nnodes)  # nnodes是节点数

    t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1])) * 0.1
    dt = t / float(iterations + 1)
    delta = np.zeros((pos.shape[0], pos.shape[0], pos.shape[1]))
    for iteration in range(iterations):
        # 点之间的差异矩阵
        delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # 每次循环初始的delta都不一样，因为后面的pos更新了
        distance = np.linalg.norm(delta,
                                  axis=-1)   # 进行范数运算：min(sum(abs(x), axis=0))，范数是对向量（或者矩阵）的度量，是一个标量（scalar）。如果axis是整数，则它指定x的轴，沿着该轴计算向量范数。
        # 强制最小距离为0.01
        np.clip(distance, 0.01, None, out=distance)  # 给定间隔，区间外的值被剪切到间隔边缘。 例如，如果指定了[0,1]的间隔，则小于0的值变为0，大于1的值变为1。
        # 位移
        displacement = np.einsum('ijk,ij->ik',  # 数组相乘，类似于矩阵点乘a.dot(b)
                                 delta,
                                 (k * k / distance ** 2 - A.T * distance / k))  # diatance中对角线的点值小，
        # update positions更新位置
        length = np.linalg.norm(displacement, axis=-1)  # 按最后一个轴的方向二范数np.sqrt(x1^2+x2^2+......+xn^2)
        length = np.where(length < 0.01, 0.1, length)  # 强制最小距离为0.1
        delta_pos = np.einsum('ij,i->ij', displacement, t / length)
        if fixed is not None:  # 如果存在需要固定的点，则重置该点的delta_pos为0
            delta_pos[fixed] = 0.0
        pos += delta_pos  # pos在原来的基础上加上位移量
        t -= dt  # t变小，每次迭代时线性降低dt
        err = np.linalg.norm(delta_pos) / nnodes  # np.linalg.norm矩阵二级范数，不保留矩阵特性，返回单个值
        if err < threshold:
            break
    return pos


##主函数#########################################################################################
# 力导向算法主体模型
def fruchterman_reingold_layout(G,
                                k=None,
                                pos=None,
                                fixed=None,
                                iterations=50,
                                threshold=1e-4,
                                weight='weight',
                                scale=1,
                                center=None,
                                dim=2):
    np.random.seed(0)

    G, center = process_params(G, center, dim)

    if fixed is not None:  # 如果标签不固定，则将标签从0,1,2....开始排序，并与原始标签一一对应
        nfixed = dict(zip(G, range(len(G))))
        fixed = np.asarray([nfixed[v] for v in fixed])

    if pos is not None:
        dom_size = max(coord for pos_tup in pos.values() for coord in pos_tup)
        if dom_size == 0:
            dom_size = 1
        pos_arr = np.random.rand(len(G), 2) * dom_size + center
        for i, n in enumerate(G):  # 将字典转换为数组
            if n in pos:  # pos是个字典，pos的索引即为点的标签
                pos_arr[i] = np.asarray(pos[n])
    else:
        pos_arr = None

    if len(G) == 0:
        return {}

    if len(G) == 1:
        return {str(G.nodes()): center}

    try:  # 点的个数大于500时使用
        if len(G) < 500:  # sparse solver for large graphs #用于大图的稀疏求解器
            raise ValueError
        A = to_scipy_sparse_matrix(G, weight=weight, dtype='f')
        if k is None and fixed is not None:
            nnodes, _ = A.shape
            k = dom_size / np.sqrt(nnodes)
        pos = sparse_fruchterman_reingold(A, k, pos_arr, fixed,
                                          iterations, threshold,
                                          dim)  # 调用函数_sparse_fruchterman_reingold
    except:  # 点的个数小于500时使用
        A = to_numpy_array(G, weight=weight)
        if k is None and fixed is not None:
            nnodes, _ = A.shape
            k = dom_size / np.sqrt(nnodes)
        pos = fruchterman_reingold(A, k, pos_arr, fixed, iterations,
                                   threshold, dim)  # 调用函数_fruchterman_reingold
    if fixed is None:
        pos = rescale_layout(pos, scale=scale) + center  # 重新缩放布局
    pos = dict(zip(G, pos))
    return pos


# 图形效果的定量分析
def score(pos):
    node_max = edge_max = 0
    node_min = edge_min = np.inf
    edge_total = edge_count = 0

    k = np.sqrt(1.0 / len(pos))
    A = to_scipy_sparse_matrix(G, weight='weight', dtype='f').tolil()
    poslist = np.array(list(pos.values()))

    for i in range(len(pos)):
        i = 1

        pos_delta = (poslist[i] - poslist).T
        posdis = np.sqrt((pos_delta ** 2).sum(axis=0))

        # 节点
        nmax_temp = posdis.max()
        nmin_temp = np.array([i for i in posdis if i != 0]).min()
        if nmax_temp > node_max:
            node_max = nmax_temp
        if nmin_temp < node_min:
            node_min = nmin_temp

        # 边
        Ai = np.asarray(A.getrowview(i).toarray())
        dege_set = np.array([posdis[j] for j in range(len(Ai[0])) if Ai[0][j] == 1])
        edge_total += dege_set.sum()
        edge_count += len(dege_set)
        emax_temp = dege_set.max()
        emin_temp = dege_set.min()
        if emax_temp > edge_max:
            edge_max = emax_temp
        if emin_temp < edge_min:
            edge_min = emin_temp

    # 节点偏差
    node_score0 = abs(k - node_min)
    node_score = '{:.2%}'.format(abs(k - node_min))

    # 边长偏差
    edge_score0 = ((edge_max - edge_total / edge_count) + (edge_total / edge_count - edge_min)) / (edge_total)
    edge_score = '{:.2%}'.format(edge_score0)

    # 总偏差
    total_score0 = node_score0 + edge_score0
    total_score = '{:.2%}'.format(total_score0)

    return node_score, edge_score, total_score


# 绘图：获取点的大小
def size_get(G, sizefile):
    size_list = []
    node_list = list(G)
    node_len = len(node_list)  # 点的个数
    size_index = dict(zip(range(node_len), node_list))
    with open(sizefile) as f:
        for line in f:
            node, size = [x for x in line.split()]
            size_list.append([int(node), float(size)])
    size_df = pd.DataFrame(size_list)
    for i in range(node_len):
        for j in range(node_len):
            if size_index[i] == size_df.iloc[j, 0]:
                size_df.iloc[j, 0] = i
    size_df = size_df.sort_values(by=0)
    size = np.array(size_df[1])
    return size_list, size


if __name__ == '__main__':
    begin = datetime.datetime.now()
    #  filename = 'D:/18.力布局图/数据.txt'
    # sizefile = 'D:/18.力布局图/size.txt'
    G = nx.Graph()

    '''
    with open(filename) as file:
        for line in file:
            # print(line.split())
            head, tail = [int(x) for x in line.split()]
            G.add_edge(head, tail)
    '''

    pos = fruchterman_reingold_layout(G)
    print(pos)
    sparse_fruchterman_reingold(G)
    '''
    size_list, size = size_get(G, sizefile)
    # size = [i*50 for i in list(size)]
    node_score, edge_score, total_score = score(pos)
    nx.draw(G, node_color='y', with_labels=True, node_size=size, width=1, edge_color='b')
    end = datetime.datetime.now()
    time_delta = end - begin
    '''
