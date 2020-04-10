import os
import random
import time
import pandas
import copy
import numpy as np
import scipy.sparse as sp


class KnowledgeGraph:
    def __init__(self, data_path, name, seed):
        self.data_path = data_path  # 数据集文件夹的路径
        self.name = name.lower()
        self.seed = seed

        # 实体
        self.entity_dict = {}  # 存储所有实体信息 {'实体1': 下标1}
        self.entity = []  # 存储所有实体的下标， 不用self.entity_dict.value() 是因为这个操作特别耗时，4800-1.5s左右
        self.n_entity = 0  # 实体的数量
        # 关系
        self.relation_dict = {}  # 存储所有关系信息 {'关系1': 下标1}
        self.n_relation = 0  # 关系数量
        # 三元组 (head, relation, tail)
        self.train_triple = []  # 训练集三元组
        self.valid_triple = []  # 验证集三元组
        self.test_triple = []  # 测试集三元组
        self.n_train_triple = 0  # 训练集数量
        self.n_valid_triple = 0  # 验证集数量
        self.n_test_triple = 0  # 测试集数量

        # 对训练集的划分
        self.fact_triple = []  # 已知的三元组（来自训练集/transE训练的三元组）
        self.n_fact_triple = 0
        self.new_triple = []  # 未知的三元组（来自训练集）
        self.n_new_triple = 0

        # ADJ
        self.sparse_one_adj = None
        # self.one_adj = None
        self.sparse_two_adj = None
        # self.two_adj = None

        # 加载dict和triple
        self.load_dict()
        self.load_triple()

        # 三元组池
        self.fact_triple_pool = set(self.fact_triple)
        self.new_triple_pool = set(self.new_triple)
        self.training_triple_pool = set(self.train_triple)
        if self.name in ['transe', 'transh', 'transd', 'transr']:
            self.golden_triple_pool = set(self.train_triple) | set(self.valid_triple) | set(self.test_triple)
        elif self.name == 'my':
            self.golden_triple_pool = set(self.train_triple) | set(self.valid_triple) | set(self.test_triple) | set(self.fact_triple)

        self.used_in_new = []
        self.triple_in_test = []

    def load_dict(self):
        # 加载实体和关系字典
        entity_path = '/entity2id.txt'
        relation_path = '/relation2id.txt'
        # 第一列为名称 第二列为id
        print("="*20 + "load entity and relation dict" + "="*20)
        entity_file = pandas.read_table(self.data_path+entity_path, header=None)
        self.entity_dict = dict(zip(entity_file[0], entity_file[1]))
        self.entity = list(self.entity_dict.values())
        self.n_entity = len(self.entity_dict)
        print("# the number of entity: {}".format(self.n_entity))

        # 关系
        relation_file = pandas.read_table(self.data_path + relation_path, header=None)
        self.relation_dict = dict(zip(relation_file[0], relation_file[1]))
        self.n_relation = len(self.relation_dict)
        print("# the number of relation: {}".format(self.n_relation))

    def load_triple(self):
        # 加载三元组(head, relation, tail)
        if self.name in ['transe', 'transh', 'transd', 'transr']:
            train_path = "/train_fact.txt"
        elif self.name == 'my':
            train_path = "/train_new.txt"
        valid_path = "/valid.txt"
        test_path = "/test.txt"
        # 加载三元组，文件格式 (head_name, tail_name, relation_name)
        print("=" * 20 + "load train/valid/test triple" + "=" * 20)

        # 训练集
        train_file = pandas.read_table(self.data_path + train_path, header=None)
        # 三元组存储下标(id) (head_id, relation_id, tail_id)
        self.train_triple = list(zip([self.entity_dict[h] for h in train_file[0]],
                                     [self.relation_dict[r] for r in train_file[2]],
                                     [self.entity_dict[t] for t in train_file[1]]))
        self.n_train_triple = len(self.train_triple)
        print("# the number of train_triple: {}".format(self.n_train_triple))

        # 验证集
        valid_file = pandas.read_table(self.data_path + valid_path, header=None)
        # 三元组存储下标(id) (head_id, relation_id, tail_id)
        self.valid_triple = list(zip([self.entity_dict[h] for h in valid_file[0]],
                                     [self.relation_dict[r] for r in valid_file[2]],
                                     [self.entity_dict[t] for t in valid_file[1]]))
        self.n_valid_triple = len(self.valid_triple)
        print("# the number of valid triple: {}".format(self.n_valid_triple))

        # 测试集
        test_file = pandas.read_table(self.data_path + test_path, header=None)
        # 三元组存储下标(id) (head_id, relation_id, tail_id)
        self.test_triple = list(zip([self.entity_dict[h] for h in test_file[0]],
                                     [self.relation_dict[r] for r in test_file[2]],
                                     [self.entity_dict[t] for t in test_file[1]]))
        self.n_test_triple = len(self.test_triple)
        print("# the number of test triple: {}".format(self.n_test_triple))
        if self.name == 'my':
            # 事实元组
            # self.fact_triple = []
            # self.n_fact_triple = len(self.fact_triple)

            fact_path = "/train_fact.txt"
            init_fact_file = pandas.read_table(self.data_path + fact_path, header=None)
            self.fact_triple = list(zip([self.entity_dict[h] for h in init_fact_file[0]],
                                         [self.relation_dict[r] for r in init_fact_file[2]],
                                         [self.entity_dict[t] for t in init_fact_file[1]]))
            self.n_fact_triple = len(self.fact_triple)
            print("# the number of fact triple: {}".format(self.n_fact_triple))

            # 新的知识 random.sample(self.train_triple, len(self.train_triple) - int(len(self.train_triple) * 0.7))
            self.new_triple = copy.deepcopy(self.train_triple)
            self.n_new_triple = len(self.new_triple)
            print("# the number of new triple: {}".format(self.n_new_triple))

            self.no_weighted_adj(self.n_entity, self.fact_triple)

    def next_batch(self, batch_size):
        """
        for transE
        """
        rand_idx = np.random.permutation(self.n_train_triple)
        start = 0
        while start < self.n_train_triple:
            end = min(start + batch_size, self.n_train_triple)
            # yield 关键字 返回一个迭代对象
            yield [self.train_triple[i] for i in rand_idx[start:end]]
            start = end

    def get_pos_neg(self, batches_triple):
        """
        for transE
        """
        batch_pos = batches_triple
        batch_neg = []
        # 随机试验 有0.5的概率为0 0.5的概率为1
        corrupt_head_prob = np.random.binomial(1, 0.5)
        for head, relation, tail in batch_pos:
            head_neg = head
            tail_neg = tail
            while True:
                if corrupt_head_prob:
                    # 如果为1 则是打乱头结点
                    # random.choice(self.entities) 从实体中随机选择一个
                    # self.entity_dict.values()是一个耗时的操作 因此提前准备好entity的数组
                    head_neg = random.choice(self.entity)
                else:
                    # 如果为0 则打乱尾结点
                    tail_neg = random.choice(self.entity)
                if (head_neg, relation, tail_neg) not in self.training_triple_pool:
                    break
            batch_neg.append((head_neg, relation, tail_neg))

        return batch_pos, batch_neg

    def write_para_vec(self, name, para, ref, text):
        print("=" * 20 + name + " write para to file" + "=" * 20)
        if ref == 'entity':
            with open('output/' + name + '/' + text + '.txt', 'w', encoding="utf-8") as file:
                for key in self.entity_dict:
                    file.write(str(self.entity_dict[key]) + '::')
                    file.write(str(key) + '::')
                    file.write(','.join(str(i) for i in para[self.entity_dict[key]]))
                    file.write("\n")
        else:
            with open('output/' + name + '/' + text + '.txt', 'w', encoding="utf-8") as file:
                for key in self.relation_dict:
                    file.write(str(self.relation_dict[key]) + '::')
                    file.write(str(key) + '::')
                    file.write(','.join(str(i) for i in para[self.relation_dict[key]]))
                    file.write("\n")

    # # for transUpdate

    def tmp_next_triple(self):
        if self.seed:
            random.seed(self.seed)
        random.shuffle(self.new_triple)
        while self.new_triple:
            print('FACT TRIPLE LEN: ', self.n_fact_triple, ' NEW TRIPLE LEN: ', self.n_new_triple)
            # yield 关键字 返回一个迭代对象
            yield [self.new_triple[0]]
            self.fact_triple.append(self.new_triple[0])
            self.used_in_new.append(self.new_triple[0])
            self.new_triple.remove(self.new_triple[0])
            self.n_fact_triple += 1
            self.n_new_triple -= 1

    @staticmethod
    def normalize_adj(adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        row_sum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(row_sum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    @staticmethod
    def sparse_to_tuple(sparse_mx):
        def to_tuple(mx):
            if not sp.isspmatrix_coo(mx):
                mx = mx.tocoo()
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
            return coords, values, shape
        if isinstance(sparse_mx, list):
            for i in range(len(sparse_mx)):
                sparse_mx[i] = to_tuple(sparse_mx[i])
        else:
            sparse_mx = to_tuple(sparse_mx)
        return sparse_mx

    def preprocess_adj(self, adj):
        """Preprocessing of adjacency matrix for simple GCN gnn and conversion to tuple representation."""
        adj_normalized = self.normalize_adj(adj + sp.eye(adj.shape[0]))  # 加上自环
        return self.sparse_to_tuple(adj_normalized)  # 返回元组模式

    def no_weighted_adj(self, total_ent_num, triple_list):
        start = time.time()
        edge = dict()
        # 获取训练集的edge矩阵
        for item in triple_list:
            if item[0] not in edge.keys():
                edge[item[0]] = dict()
            if item[2] not in edge.keys():
                edge[item[2]] = dict()
            edge[item[0]][item[2]] = item[1]
            edge[item[2]][item[0]] = item[1]
            if item[1] == 0:
                edge[item[0]][item[2]] = -2
                edge[item[2]][item[0]] = -2

        # edge[5000] = dict()
        # edge[5000].setdefault(2000, 5)
        # edge[5000].setdefault(1000, 2)
        row = list()
        col = list()
        data = list()
        for i in range(total_ent_num):
            if i not in edge.keys():
                continue
            key = i  # 训练集中的一个node
            value = edge[key].keys()  # 该node邻接的node
            add_key_len = len(value)  # 该node邻接node的个数
            add_key = (key * np.ones(add_key_len)).tolist()  # ex. 2, 2, 2
            row.extend(add_key)  # 大图中的横坐标
            col.extend(list(value))  # 大图中的纵坐标
            data.extend(list(edge[key].values()))
        # data_len = len(row)
        # data = np.ones(data_len)  # 数据都为1表示邻接
        self.sparse_one_adj = sp.coo_matrix((data, (row, col)), shape=(total_ent_num, total_ent_num))
        # print('$$$', self.sparse_one_adj.getrow(32626))
        # self.one_adj = self.preprocess_adj(self.sparse_one_adj)

        expend_edge = dict()
        row = list()
        col = list()
        temp_len = 0
        for key, values in edge.items():
            if key not in expend_edge.keys():
                expend_edge[key] = set()
            for value in values:
                add_value = edge[value]
                for item in add_value:
                    if item not in values and item != key:
                        expend_edge[key].add(item)
                        no_len = len(expend_edge[key])
                        if temp_len != no_len:
                            row.append(key)
                            col.append(item)
                        temp_len = no_len
        data = np.ones(len(row))
        self.sparse_two_adj = sp.coo_matrix((data, (row, col)), shape=(total_ent_num, total_ent_num))
        # self.two_adj = self.preprocess_adj(self.sparse_two_adj)
        print('ADJ GENERATED COST: {:.4f}s.'.format(time.time() - start))

    def find_used_in_test(self):
        node_set = set()
        for i in self.used_in_new:
            node_set.add(i[0])
            node_set.add(i[2])
        for i in self.test_triple:
            if i[0] in node_set or i[2] in node_set:
                self.triple_in_test.append(i)
        self.triple_in_test = list(set(self.triple_in_test))
        # print('@', node_set)
        # print('x', self.triple_in_test)

    def result2txt(self, name, cmp_name, raw_d, filter_d):
        """
        'new_triple': n_new_triple, 'used_eval_triple': n_used_eval_triple,
        'H_MR': head_meanrank_raw, 'H_h10': head_hits10_raw,
        'T_MR': tail_meanrank_raw, 'T_h10': tail_hits10_raw,
        'AVE_MR': (head_meanrank_raw + tail_meanrank_raw) / 2,
        'AVE_h10': (head_hits10_raw + tail_hits10_raw) / 2
        """
        with open('output/' + cmp_name + '/' + name + '_raw.txt', 'a', encoding='utf-8') as file:
            file.write(str(raw_d['new_triple']) + ',' + str(raw_d['used_eval_triple']) + ',' +
                       str(raw_d['H_MR']) + ',' + str(raw_d['H_h10']) + ',' +
                       str(raw_d['T_MR']) + ',' + str(raw_d['T_h10']) + ',' +
                       str(raw_d['AVE_MR']) + ',' + str(raw_d['AVE_h10']) + '\n')
        with open('output/' + cmp_name + '/' + name + '_filter.txt', 'a', encoding='utf-8') as file:
            file.write(str(filter_d['new_triple']) + ',' + str(filter_d['used_eval_triple']) + ',' +
                       str(filter_d['H_MR']) + ',' + str(filter_d['H_h10']) + ',' +
                       str(filter_d['T_MR']) + ',' + str(filter_d['T_h10']) + ',' +
                       str(filter_d['AVE_MR']) + ',' + str(filter_d['AVE_h10']) + '\n')
