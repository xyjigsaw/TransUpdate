# Name: test3
# Author: Reacubeth
# Time: 2020/3/27 18:15
# Mail: noverfitting@gmail.com
# Site: www.omegaxyz.com
# *_*coding:utf-8 *_*

import numpy as np
from scipy.sparse import coo_matrix


def sparse_fruchterman_reingold(A, k=None, pos=None, fixed=None, iterations=50, threshold=1e-5, dim=3):
    # np.random.seed(1)
    nodes_num = A.shape[0]
    A = A.tolil()
    A = A.astype('float')
    if pos is None:
        pos = np.asarray(np.random.rand(nodes_num, dim), dtype=A.dtype)
        print('Init pos', pos)
    else:
        pos = np.array(pos)
        pos = pos.astype(A.dtype)

    if fixed is None:
        fixed = []

    if k is None:
        k = np.sqrt(1.0 / nodes_num)
    t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1])) * 0.1
    dt = t / float(iterations + 1)
    displacement = np.zeros((dim, nodes_num))
    for iteration in range(iterations):
        displacement *= 0  # 数组的形状不变，每个值都为0
        for i in range(A.shape[0]):  # 行数
            if i in fixed:  # fixed是数组，保存固定不变的坐标的索引，若i在索引中，则跳过以下步骤
                continue
            delta = (pos[i] - pos).T  # pos[0]是第零行，pos[0]是一行，用于减去pos的每一行
            distance = np.sqrt((delta ** 2).sum(axis=0))  # delta**2是数组中的每个值都平方;axis=0每一列相加
            distance = np.where(distance < 0.01, 0.01, distance)
            Ai = np.asarray(A.getrowview(i).toarray())
            print('Ai', Ai)
            displacement[:, i] += \
                (delta * (k * k / distance ** 2 - Ai * distance / k)).sum(axis=1)
        # update positions 更新位置
        length = np.sqrt((displacement ** 2).sum(axis=0))
        length = np.where(length < 0.01, 0.1, length)
        delta_pos = (displacement * t / length).T
        pos += delta_pos
        # cool temperature 冷却温度
        t -= dt
        err = np.linalg.norm(delta_pos) / nodes_num
        if err < threshold:
            break

    return pos


if __name__ == '__main__':

    row = np.array([0, 0, 1, 2, 3, 3])
    col = np.array([1, 3, 0, 3, 0, 2])
    data = np.array([1, 1, 1, 1, 1, 1])
    coo_matrix((data, (row, col)), shape=(4, 4)).toarray()
    print(coo_matrix((data, (row, col)), shape=(4, 4)))
    a = sparse_fruchterman_reingold(coo_matrix((data, (row, col)), shape=(4, 4)))
    print(a)


'''
    def force(self, fixed=None, k=None):
        nodes_num = self.entity_embedding.shape[0]
        A = self.sparse_adj.tolil()
        if fixed is None:
            fixed = []
        if k is None:
            k = tf.sqrt(1.0 / nodes_num)
        t = 0.5
        dt = tf.constant(t / 2.0)

        displacement = tf.zeros([self.embedding_dim, nodes_num])
        for i in range(nodes_num):
            if i in fixed:
                continue
            delta = tf.slice(self.entity_embedding, [i, 0], [1, -1]) - self.entity_embedding
            distance = tf.sqrt(tf.reduce_sum(tf.square(delta), axis=0))
            # distance = tf.where(distance < 0.01, 0.01, distance)
            Ai = tf.convert_to_tensor(np.asarray(A.getrowview(i).toarray()))
            displacement
        self.entity_embedding = self.entity_embedding
'''
