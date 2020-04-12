# Name: transuttnew
# Author: Reacubeth
# Time: 2020/4/12 10:30
# Mail: noverfitting@gmail.com
# Site: www.omegaxyz.com
# *_*coding:utf-8 *_*

import timeit
import tensorflow.compat.v1 as tf
from KG_data import KnowledgeGraph
import numpy as np

tf.compat.v1.disable_eager_execution()


class TransUpdate:
    def __init__(self, cmp_name, kg: KnowledgeGraph, dissimilarity_func, learning_rate, epoch, eval_times):
        # 根据文献 lr = {0.001, 0.01, 0.1} embedding_dim = {20, 50}
        # WN18数据集 lr = 0.01  dim = 50  d_f = L1
        # FB15K lr = 0.01  dim = 50  d_f = L2
        self.cmp_name = cmp_name.lower()
        self.epoch = epoch
        self.eval_times = eval_times

        self.kg = kg  # 知识图谱三元组
        self.dissimilarity_func = dissimilarity_func  # 不相似函数(稀疏函数) 一般取 L1 或 L2
        self.learning_rate = learning_rate  # 学习率
        # 初始化一个三元组
        self.triple = tf.placeholder(dtype=tf.int32, shape=[None, 3])

        self.sparse_neighbor = tf.placeholder(dtype=tf.int32)
        self.sparse_neighbor_noself = tf.placeholder(dtype=tf.int32)
        self.prob_e = tf.placeholder(dtype=tf.float32)

        self.sparse_neighbor_tail = tf.placeholder(dtype=tf.int32)
        self.sparse_neighbor_noself_tail = tf.placeholder(dtype=tf.int32)
        self.prob_e_tail = tf.placeholder(dtype=tf.float32)

        self.train_op = None  # 训练操作，是最小化优化器的那一步 如 tf.train.AdamOptimizer().minimize()
        self.loss = None  # 损失函数
        self.global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')  # 全局训练步数

        # 初始embedding
        self.entity_embedding = None
        self.relation_embedding = None
        self.embedding_dim = None

        # online评估操作
        self.eval_triple = tf.placeholder(dtype=tf.int32, shape=[None, 3])  # 评估三元组，3行n列
        self.idx_head_prediction = None
        self.idx_tail_prediction = None

        # transX评估操作
        self.eval_triple4raw = tf.placeholder(dtype=tf.int32, shape=[None, 3])  # 评估三元组，3行n列
        self.idx_head_prediction4raw = None
        self.idx_tail_prediction4raw = None
        self.entity_embedding4raw = None
        self.relation_embedding4raw = None
        # transD
        self.ent_transfer4d = None
        self.rel_transfer4d = None
        # transH
        self.normal_vector4h = None
        # transR
        self.hidden_sizeE4r = self.embedding_dim
        self.hidden_sizeR4r = self.embedding_dim
        self.rel_matrix4r = None

        self.load_embedding()  # 读取初始词嵌入和维度
        self.load_embedding4raw()
        self.init_train()  # 初始化训练步骤
        self.build_eval_graph()
        self.build_eval_graph4raw()

    def load_embedding(self):
        """
        读取已有嵌入模型
        """
        start = timeit.default_timer()
        f = open('output/' + self.cmp_name + '/entity_embedding.txt', 'r')
        line = f.readline().strip()
        entity_embedding = []
        while line:
            if '::' in line:
                e_id, e, raw = line.split('::')
                embedding = [float(i) for i in raw.split(',')]
                entity_embedding.append(embedding)
            line = f.readline().strip()
        f.close()
        self.entity_embedding = tf.Variable(tf.convert_to_tensor(entity_embedding), name="entity_embedding")
        self.embedding_dim = self.entity_embedding.get_shape()[1]
        del entity_embedding
        self.hidden_sizeE4r = self.embedding_dim
        self.hidden_sizeR4r = self.embedding_dim

        f = open('output/' + self.cmp_name + '/relation_embedding.txt', 'r')
        line = f.readline()
        relation_embedding = []
        while line:
            if '::' in line:
                e_id, e, raw = line.split('::')
                embedding = [float(i) for i in raw.split(',')]
                relation_embedding.append(embedding)
            line = f.readline().strip()
        f.close()
        self.relation_embedding = tf.Variable(tf.convert_to_tensor(relation_embedding), name="relation_embedding")
        del relation_embedding

        print('LOADING EMB COST TIME: {:.4f}s.'.format(timeit.default_timer() - start))

    def init_train(self):
        # 正向传播 计算损失 训练步骤
        # 正则化
        self.entity_embedding = tf.nn.l2_normalize(self.entity_embedding, dim=1)
        self.relation_embedding = tf.nn.l2_normalize(self.relation_embedding, dim=1)

        p_neighbor = self.get_prob(self.triple, self.sparse_neighbor, 0)
        p_real_neighbor = self.get_real_prob(self.sparse_neighbor_noself)

        p_neighbor_tail = self.get_prob(self.triple, self.sparse_neighbor_tail, 2)
        p_real_neighbor_tail = self.get_real_prob(self.sparse_neighbor_noself_tail)

        # para1 = tf.shape(self.sparse_neighbor)[0]
        # para2 = tf.shape(self.sparse_neighbor_tail)[0]
        # sum_para = tf.add(para1, para2)
        # para1 = tf.cast(tf.divide(para1, sum_para), dtype=tf.float32)
        # para2 = tf.cast(tf.divide(para2, sum_para), dtype=tf.float32)

        # distance = p_neighbor - p_real_neighbor
        self.loss = tf.reduce_mean(-tf.reduce_sum(tf.log(p_neighbor) * p_real_neighbor, axis=1)) + \
                    tf.reduce_mean(-tf.reduce_sum(tf.log(p_neighbor_tail) * p_real_neighbor_tail, axis=1))

        self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate). \
            minimize(self.loss, global_step=self.global_step)

    def get_prob(self, triple, sparse_neighbor, pos):
        vec = tf.nn.embedding_lookup(self.entity_embedding, triple[:, pos])
        # p_h = tf.Variable(tf.zeros([1, self.kg.n_entity]), name="p_h")
        v_neighbors = tf.nn.embedding_lookup(self.entity_embedding, sparse_neighbor)
        sum_neighbor_mat = tf.exp(tf.matmul(v_neighbors, tf.transpose(vec)))
        p_neighbor = tf.divide(sum_neighbor_mat, tf.reduce_sum(sum_neighbor_mat))  # neighbor_num * 1
        # print('sum_neighbor', sum_neighbor_mat)
        return tf.transpose(p_neighbor)

    def get_real_prob(self, sparse_neighbor_noself):
        h1_neighbors = tf.nn.embedding_lookup(self.entity_embedding, sparse_neighbor_noself)
        new_emb = [tf.reduce_mean(h1_neighbors, 0)]  # neighbor的均值即new emb

        v_neighbors = tf.nn.embedding_lookup(self.entity_embedding, sparse_neighbor_noself)
        v_neighbors = tf.concat([v_neighbors, new_emb], 0)
        sum_neighbor_mat = tf.exp(tf.matmul(v_neighbors, tf.transpose(new_emb)))
        p_real_neighbor = tf.divide(sum_neighbor_mat, tf.reduce_sum(sum_neighbor_mat))
        return tf.transpose(p_real_neighbor)

    def launch_training(self, sess):
        # 开始训练
        start = timeit.default_timer()

        n_used_triple = 0
        for triple in self.kg.tmp_next_triple():
            if n_used_triple % self.eval_times == 0:
                try:
                    self.launch_evaluation(sess, n_used_triple)
                    self.launch_evaluation4raw(sess, n_used_triple)
                except ZeroDivisionError as e:
                    print('ERROR:', n_used_triple)
            n_used_triple += 1
            print("=" * 20 + "TRAINING FACT {}".format(n_used_triple) + "=" * 20)
            print('triple: ', triple)
            #  head
            neighbor_id_rel = dict()
            rel_cnt = dict()
            for i in str(self.kg.sparse_one_adj.getrow(triple[0][0]).tolil()).split('\n'):
                i = str(i).strip('\n')
                try:
                    rel_id = int(i[i.index('\t') + 1:])
                    if rel_id == -2:
                        rel_id = 0
                    neighbor_id_rel[int(i[i.index(', ') + 2:i.index(')')])] = rel_id
                    if rel_id not in rel_cnt.keys():
                        rel_cnt[rel_id] = 0
                    rel_cnt[rel_id] += 1
                except Exception as e:
                    pass
            neighbor_id_rel[triple[0][2]] = triple[0][1]
            rel_cnt[triple[0][1]] = 1
            neighbor_id_rel[triple[0][0]] = -1
            rel_cnt[-1] = 1
            prob_e = []
            for i in list(neighbor_id_rel.keys()):
                prob_e.append(rel_cnt[neighbor_id_rel[i]])

            prob_e = np.array(prob_e) / sum(list(rel_cnt.values()))
            # print('neighbor_id_rel:', neighbor_id_rel)

            #  head_tail
            neighbor_id_rel_tail = dict()
            rel_cnt_tail = dict()
            for i in str(self.kg.sparse_one_adj.getrow(triple[0][2]).tolil()).split('\n'):
                i = str(i).strip('\n')
                try:
                    rel_id = int(i[i.index('\t') + 1:])
                    if rel_id == -2:
                        rel_id = 0
                    neighbor_id_rel_tail[int(i[i.index(', ') + 2:i.index(')')])] = rel_id
                    if rel_id not in rel_cnt_tail.keys():
                        rel_cnt_tail[rel_id] = 0
                    rel_cnt_tail[rel_id] += 1
                except Exception as e:
                    pass
            neighbor_id_rel_tail[triple[0][0]] = triple[0][1]
            rel_cnt_tail[triple[0][1]] = 1
            neighbor_id_rel_tail[triple[0][2]] = -1
            rel_cnt_tail[-1] = 1
            prob_e_tail = []
            for i in list(neighbor_id_rel_tail.keys()):
                prob_e_tail.append(rel_cnt_tail[neighbor_id_rel_tail[i]])

            prob_e_tail = np.array(prob_e_tail) / sum(list(rel_cnt_tail.values()))
            # print('neighbor_id_rel_tail:', neighbor_id_rel_tail)

            epoch_loss = 0
            for i in range(self.epoch):
                regret_loss, _ = sess.run(fetches=[self.loss, self.train_op],
                                          feed_dict={self.triple: triple,
                                                     self.sparse_neighbor: np.array(list(neighbor_id_rel.keys())),
                                                     self.sparse_neighbor_noself: np.array(
                                                         list(neighbor_id_rel.keys())[:-1]),
                                                     self.prob_e: prob_e,

                                                     self.sparse_neighbor_tail:
                                                         np.array(list(neighbor_id_rel_tail.keys())),
                                                     self.sparse_neighbor_noself_tail:
                                                         np.array(list(neighbor_id_rel_tail.keys())[:-1]),
                                                     self.prob_e_tail: prob_e_tail})
                # print('regret_loss', regret_loss)
                epoch_loss += regret_loss
                # print('REGRET: {:.5f}'.format(regret_loss))
            print('[TRAIN {:.3f}s] #triple: {}/{} Done.'.format(timeit.default_timer() - start,
                                                                n_used_triple,
                                                                self.kg.n_train_triple))
            print('EPOCH LOSS (all regret single fact): {:.3f}'.format(epoch_loss))
            print('EMB:', np.array(self.entity_embedding.eval()).sum())
            self.kg.sparse_one_adj = None
            self.kg.sparse_two_adj = None
            self.kg.no_weighted_adj(self.kg.n_entity, self.kg.fact_triple)

    def build_eval_graph(self):
        with tf.name_scope('evaluation'):
            self.idx_head_prediction, self.idx_tail_prediction = self.evaluate(self.eval_triple)

    def evaluate(self, eval_triple):
        with tf.name_scope('lookup'):
            head = tf.nn.embedding_lookup(self.entity_embedding, eval_triple[:, 0])  # 头结点
            tail = tf.nn.embedding_lookup(self.entity_embedding, eval_triple[:, 2])  # 尾结点
            relation = tf.nn.embedding_lookup(self.relation_embedding, eval_triple[:, 1])  # 关系

        with tf.name_scope('link'):
            if self.cmp_name == 'transe':
                print('TransE Eva.')
                # 头结点预测 通过训练的节点嵌入来代替原始头结点
                distance_head_prediction = self.entity_embedding + relation - tail
                # 尾结点预测 通过训练的节点嵌入来代替原始尾结点
                distance_tail_prediction = head + relation - self.entity_embedding
            elif self.cmp_name == 'transd':
                print('TransD Eva.')
                head_transd = tf.nn.embedding_lookup(self.ent_transfer4d, eval_triple[:, 0])
                rel_transd = tf.nn.embedding_lookup(self.rel_transfer4d, eval_triple[:, 1])
                tail_transd = tf.nn.embedding_lookup(self.ent_transfer4d, eval_triple[:, 2])

                all_emb = self.cal4transD(self.entity_embedding, self.ent_transfer4d,
                                          tf.tile(rel_transd, [self.kg.n_entity, 1]))
                new_head = self.cal4transD(head, head_transd, rel_transd)
                new_tail = self.cal4transD(tail, tail_transd, rel_transd)

                distance_head_prediction = all_emb + relation - new_tail
                distance_tail_prediction = new_head + relation - all_emb
            elif self.cmp_name == 'transh':
                print('TransH Eva.')
                norm = tf.nn.embedding_lookup(self.normal_vector4h, eval_triple[:, 1])

                all_emb = self.cal4transH(self.entity_embedding, tf.tile(norm, [self.kg.n_entity, 1]))
                new_head = self.cal4transH(head, norm)
                new_tail = self.cal4transH(tail, norm)

                distance_head_prediction = all_emb + relation - new_tail
                distance_tail_prediction = new_head + relation - all_emb
            elif self.cmp_name == 'transr':
                print('TransR Eva.')
                all_emb = tf.reshape(self.entity_embedding, [-1, self.hidden_sizeE4r, 1])
                new_head = tf.reshape(head, [-1, self.hidden_sizeE4r, 1])
                new_tail = tf.reshape(tail, [-1, self.hidden_sizeE4r, 1])
                new_rel = tf.reshape(relation, [-1, self.hidden_sizeR4r])

                transr_matrix = tf.reshape(tf.nn.embedding_lookup(self.rel_matrix4r, eval_triple[:, 1]),
                                           [-1, self.hidden_sizeR4r, self.hidden_sizeE4r])

                all_emb = tf.nn.l2_normalize(tf.reshape(tf.matmul(transr_matrix, all_emb),
                                                        [-1, self.hidden_sizeR4r]), 1)
                new_head = tf.nn.l2_normalize(tf.reshape(tf.matmul(transr_matrix, new_head),
                                                         [-1, self.hidden_sizeR4r]), 1)
                new_tail = tf.nn.l2_normalize(tf.reshape(tf.matmul(transr_matrix, new_tail),
                                                         [-1, self.hidden_sizeR4r]), 1)

                distance_head_prediction = all_emb + new_rel - new_tail
                distance_tail_prediction = new_head + new_rel - all_emb

        with tf.name_scope('rank'):
            if self.dissimilarity_func == 'L1':  # L1 score
                # tf.nn.top_k 返回 input 中每行最大的 k 个数，并且返回它们所在位置的索引
                _, idx_head_prediction = tf.nn.top_k(tf.reduce_sum(tf.abs(distance_head_prediction), axis=1),
                                                     k=self.kg.n_entity)
                _, idx_tail_prediction = tf.nn.top_k(tf.reduce_sum(tf.abs(distance_tail_prediction), axis=1),
                                                     k=self.kg.n_entity)
            else:  # L2 score
                _, idx_head_prediction = tf.nn.top_k(tf.reduce_sum(tf.square(distance_head_prediction), axis=1),
                                                     k=self.kg.n_entity)
                _, idx_tail_prediction = tf.nn.top_k(tf.reduce_sum(tf.square(distance_tail_prediction), axis=1),
                                                     k=self.kg.n_entity)
        return idx_head_prediction, idx_tail_prediction

    def launch_evaluation(self, session, n_new_triple):
        print("=" * 20 + "EVALUATION" + "=" * 20)
        start = timeit.default_timer()
        rank_result_queue = []
        n_used_eval_triple = 0
        self.kg.find_used_in_test()

        for eval_triple in self.kg.triple_in_test:
            idx_head_prediction, idx_tail_prediction = session.run(fetches=[self.idx_head_prediction,
                                                                            self.idx_tail_prediction],
                                                                   feed_dict={self.eval_triple: [eval_triple]})
            eval_result_queue = [eval_triple, idx_head_prediction, idx_tail_prediction]
            rank_result_queue.append(eval_result_queue)
            n_used_eval_triple += 1
            if n_used_eval_triple % 100 == 0:
                print('[EVA {:.3f}s] #evaluation triple: {}/{}'.format(timeit.default_timer() - start,
                                                                       n_used_eval_triple,
                                                                       len(self.kg.triple_in_test)))
        '''Raw'''
        head_meanrank_raw = 0
        head_hits10_raw = 0
        tail_meanrank_raw = 0
        tail_hits10_raw = 0
        '''Filter'''
        head_meanrank_filter = 0
        head_hits10_filter = 0
        tail_meanrank_filter = 0
        tail_hits10_filter = 0
        for i in range(n_used_eval_triple):
            head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter = self.calculate_rank(rank_result_queue[i])
            head_meanrank_raw += head_rank_raw
            if head_rank_raw < 10:
                head_hits10_raw += 1
            tail_meanrank_raw += tail_rank_raw
            if tail_rank_raw < 10:
                tail_hits10_raw += 1
            head_meanrank_filter += head_rank_filter
            if head_rank_filter < 10:
                head_hits10_filter += 1
            tail_meanrank_filter += tail_rank_filter
            if tail_rank_filter < 10:
                tail_hits10_filter += 1
        print('-----Raw-----')
        head_meanrank_raw /= n_used_eval_triple
        head_hits10_raw /= n_used_eval_triple
        tail_meanrank_raw /= n_used_eval_triple
        tail_hits10_raw /= n_used_eval_triple
        print('-----Head prediction-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(head_meanrank_raw, head_hits10_raw))
        print('-----Tail prediction-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(tail_meanrank_raw, tail_hits10_raw))
        print('------Average------')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format((head_meanrank_raw + tail_meanrank_raw) / 2,
                                                         (head_hits10_raw + tail_hits10_raw) / 2))
        head_meanrank_filter /= n_used_eval_triple
        head_hits10_filter /= n_used_eval_triple
        tail_meanrank_filter /= n_used_eval_triple
        tail_hits10_filter /= n_used_eval_triple
        raw_res_dict = {'new_triple': n_new_triple, 'used_eval_triple': n_used_eval_triple,
                        'H_MR': head_meanrank_raw, 'H_h10': head_hits10_raw,
                        'T_MR': tail_meanrank_raw, 'T_h10': tail_hits10_raw,
                        'AVE_MR': (head_meanrank_raw + tail_meanrank_raw) / 2,
                        'AVE_h10': (head_hits10_raw + tail_hits10_raw) / 2}
        filter_res_dict = {'new_triple': n_new_triple, 'used_eval_triple': n_used_eval_triple,
                           'H_MR': head_meanrank_filter, 'H_h10': head_hits10_filter,
                           'T_MR': tail_meanrank_filter, 'T_h10': tail_hits10_filter,
                           'AVE_MR': (head_meanrank_filter + tail_meanrank_filter) / 2,
                           'AVE_h10': (head_hits10_filter + tail_hits10_filter) / 2}
        self.kg.result2txt('my', self.cmp_name, raw_res_dict, filter_res_dict)

    def calculate_rank(self, idx_predictions):
        eval_triple, idx_head_prediction, idx_tail_prediction = idx_predictions
        head, relation, tail = eval_triple
        head_rank_raw = 0
        tail_rank_raw = 0
        head_rank_filter = 0
        tail_rank_filter = 0
        # idx_head_prediction[::-1] 倒序复制一遍 之前是从大到小 倒序后从小到大
        for candidate in idx_head_prediction[::-1]:
            if candidate == head:
                break
            else:
                head_rank_raw += 1
                if (candidate, relation, tail) in self.kg.golden_triple_pool:
                    continue
                else:
                    head_rank_filter += 1
        for candidate in idx_tail_prediction[::-1]:
            if candidate == tail:
                break
            else:
                tail_rank_raw += 1
                if (head, relation, candidate) in self.kg.golden_triple_pool:
                    continue
                else:
                    tail_rank_filter += 1
        return head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter

    ######## eva4

    def cal4transD(self, e, t, r):
        return tf.nn.l2_normalize(e + tf.reduce_sum(e * t, 1, keep_dims=True) * r, 1)

    def cal4transH(self, e, n):
        norm = tf.nn.l2_normalize(n, 1)
        return e - tf.reduce_sum(e * norm, 1, keep_dims=True) * norm

    def load_embedding4raw(self):
        """
        读取已有嵌入模型
        """
        start = timeit.default_timer()
        f = open('output/' + self.cmp_name + '/entity_embedding.txt', "r")
        line = f.readline().strip()
        entity_embedding = []
        while line:
            if '::' in line:
                e_id, e, raw = line.split('::')
                embedding = [float(i) for i in raw.split(',')]
                entity_embedding.append(embedding)
            line = f.readline().strip()
        f.close()
        self.entity_embedding4raw = tf.Variable(tf.convert_to_tensor(entity_embedding))
        del entity_embedding

        f = open('output/' + self.cmp_name + '/relation_embedding.txt', "r")
        line = f.readline()
        relation_embedding = []
        while line:
            if '::' in line:
                e_id, e, raw = line.split('::')
                embedding = [float(i) for i in raw.split(',')]
                relation_embedding.append(embedding)
            line = f.readline().strip()
        f.close()
        self.relation_embedding4raw = tf.Variable(tf.convert_to_tensor(relation_embedding))
        del relation_embedding

        if self.cmp_name == 'transe':
            pass
        elif self.cmp_name == 'transd':
            f = open('output/' + self.cmp_name + '/ent_transfer4d.txt', 'r')
            line = f.readline().strip()
            para = []
            while line:
                if '::' in line:
                    e_id, e, raw = line.split('::')
                    item = [float(i) for i in raw.split(',')]
                    para.append(item)
                line = f.readline().strip()
            f.close()
            self.ent_transfer4d = tf.Variable(tf.convert_to_tensor(para), name="ent_transfer4d")
            del para

            f = open('output/' + self.cmp_name + '/rel_transfer4d.txt', 'r')
            line = f.readline()
            para = []
            while line:
                if '::' in line:
                    e_id, e, raw = line.split('::')
                    item = [float(i) for i in raw.split(',')]
                    para.append(item)
                line = f.readline().strip()
            f.close()
            self.rel_transfer4d = tf.Variable(tf.convert_to_tensor(para), name="rel_transfer4d")
            del para
        elif self.cmp_name == 'transh':
            f = open('output/' + self.cmp_name + '/normal_vector4h.txt', 'r')
            line = f.readline()
            para = []
            while line:
                if '::' in line:
                    e_id, e, raw = line.split('::')
                    item = [float(i) for i in raw.split(',')]
                    para.append(item)
                line = f.readline().strip()
            f.close()
            self.normal_vector4h = tf.Variable(tf.convert_to_tensor(para), name="normal_vector4h")
            del para
        elif self.cmp_name == 'transr':
            f = open('output/' + self.cmp_name + '/rel_matrix4r.txt', 'r')
            line = f.readline()
            para = []
            while line:
                if '::' in line:
                    e_id, e, raw = line.split('::')
                    item = [float(i) for i in raw.split(',')]
                    para.append(item)
                line = f.readline().strip()
            f.close()
            self.rel_matrix4r = tf.Variable(tf.convert_to_tensor(para), name="rel_matrix4r")
            del para

        print('LOADING ' + self.cmp_name + ' PARA COST TIME: {:.4f}s.'.format(timeit.default_timer() - start))

    def build_eval_graph4raw(self):
        with tf.name_scope('evaluation4raw'):
            self.idx_head_prediction4raw, self.idx_tail_prediction4raw = self.evaluate4raw(self.eval_triple4raw)

    def evaluate4raw(self, eval_triple4raw):
        with tf.name_scope('lookup4raw'):
            head = tf.nn.embedding_lookup(self.entity_embedding4raw, eval_triple4raw[:, 0])  # 头结点
            tail = tf.nn.embedding_lookup(self.entity_embedding4raw, eval_triple4raw[:, 2])  # 尾结点
            relation = tf.nn.embedding_lookup(self.relation_embedding4raw, eval_triple4raw[:, 1])  # 关系

        with tf.name_scope('link4raw'):
            if self.cmp_name == 'transe':
                print('TransE Eva.')
                # 头结点预测 通过训练的节点嵌入来代替原始头结点
                distance_head_prediction = self.entity_embedding4raw + relation - tail
                # 尾结点预测 通过训练的节点嵌入来代替原始尾结点
                distance_tail_prediction = head + relation - self.entity_embedding4raw
            elif self.cmp_name == 'transd':
                print('TransD Eva.')
                head_transd = tf.nn.embedding_lookup(self.ent_transfer4d, eval_triple4raw[:, 0])
                rel_transd = tf.nn.embedding_lookup(self.rel_transfer4d, eval_triple4raw[:, 1])
                tail_transd = tf.nn.embedding_lookup(self.ent_transfer4d, eval_triple4raw[:, 2])

                all_emb = self.cal4transD(self.entity_embedding4raw, self.ent_transfer4d,
                                          tf.tile(rel_transd, [self.kg.n_entity, 1]))
                new_head = self.cal4transD(head, head_transd, rel_transd)
                new_tail = self.cal4transD(tail, tail_transd, rel_transd)

                distance_head_prediction = all_emb + relation - new_tail
                distance_tail_prediction = new_head + relation - all_emb
            elif self.cmp_name == 'transh':
                print('TransH Eva.')
                norm = tf.nn.embedding_lookup(self.normal_vector4h, eval_triple4raw[:, 1])

                all_emb = self.cal4transH(self.entity_embedding4raw, tf.tile(norm, [self.kg.n_entity, 1]))
                new_head = self.cal4transH(head, norm)
                new_tail = self.cal4transH(tail, norm)

                distance_head_prediction = all_emb + relation - new_tail
                distance_tail_prediction = new_head + relation - all_emb
            elif self.cmp_name == 'transr':
                print('TransR Eva.')
                all_emb = tf.reshape(self.entity_embedding4raw, [-1, self.hidden_sizeE4r, 1])
                new_head = tf.reshape(head, [-1, self.hidden_sizeE4r, 1])
                new_tail = tf.reshape(tail, [-1, self.hidden_sizeE4r, 1])
                new_rel = tf.reshape(relation, [-1, self.hidden_sizeR4r])

                transr_matrix = tf.reshape(tf.nn.embedding_lookup(self.rel_matrix4r, eval_triple4raw[:, 1]),
                                           [-1, self.hidden_sizeR4r, self.hidden_sizeE4r])

                all_emb = tf.nn.l2_normalize(tf.reshape(tf.matmul(transr_matrix, all_emb),
                                                        [-1, self.hidden_sizeR4r]), 1)
                new_head = tf.nn.l2_normalize(tf.reshape(tf.matmul(transr_matrix, new_head),
                                                         [-1, self.hidden_sizeR4r]), 1)
                new_tail = tf.nn.l2_normalize(tf.reshape(tf.matmul(transr_matrix, new_tail),
                                                         [-1, self.hidden_sizeR4r]), 1)

                distance_head_prediction = all_emb + new_rel - new_tail
                distance_tail_prediction = new_head + new_rel - all_emb

        with tf.name_scope('rank4raw'):
            if self.dissimilarity_func == 'L1':  # L1 score
                # tf.nn.top_k 返回 input 中每行最大的 k 个数，并且返回它们所在位置的索引
                _, idx_head_prediction = tf.nn.top_k(tf.reduce_sum(tf.abs(distance_head_prediction), axis=1),
                                                     k=self.kg.n_entity)
                _, idx_tail_prediction = tf.nn.top_k(tf.reduce_sum(tf.abs(distance_tail_prediction), axis=1),
                                                     k=self.kg.n_entity)
            else:  # L2 score
                _, idx_head_prediction = tf.nn.top_k(tf.reduce_sum(tf.square(distance_head_prediction), axis=1),
                                                     k=self.kg.n_entity)
                _, idx_tail_prediction = tf.nn.top_k(tf.reduce_sum(tf.square(distance_tail_prediction), axis=1),
                                                     k=self.kg.n_entity)
        return idx_head_prediction, idx_tail_prediction

    def launch_evaluation4raw(self, session, n_new_triple):
        print("=" * 20 + "EVALUATION4raw" + "=" * 20)
        start = timeit.default_timer()
        rank_result_queue = []
        n_used_eval_triple = 0

        for eval_triple4raw in self.kg.triple_in_test:
            idx_head_prediction, idx_tail_prediction = session.run(fetches=[self.idx_head_prediction4raw,
                                                                            self.idx_tail_prediction4raw],
                                                                   feed_dict={self.eval_triple4raw: [eval_triple4raw]})
            eval_result_queue = [eval_triple4raw, idx_head_prediction, idx_tail_prediction]
            rank_result_queue.append(eval_result_queue)
            n_used_eval_triple += 1
            if n_used_eval_triple % 100 == 0:
                print('[EVA {:.3f}s] #evaluation triple: {}/{}'.format(timeit.default_timer() - start,
                                                                       n_used_eval_triple,
                                                                       len(self.kg.triple_in_test)))
        '''Raw'''
        head_meanrank_raw = 0
        head_hits10_raw = 0
        tail_meanrank_raw = 0
        tail_hits10_raw = 0
        '''Filter'''
        head_meanrank_filter = 0
        head_hits10_filter = 0
        tail_meanrank_filter = 0
        tail_hits10_filter = 0
        for i in range(n_used_eval_triple):
            head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter = self.calculate_rank(rank_result_queue[i])
            head_meanrank_raw += head_rank_raw
            if head_rank_raw < 10:
                head_hits10_raw += 1
            tail_meanrank_raw += tail_rank_raw
            if tail_rank_raw < 10:
                tail_hits10_raw += 1
            head_meanrank_filter += head_rank_filter
            if head_rank_filter < 10:
                head_hits10_filter += 1
            tail_meanrank_filter += tail_rank_filter
            if tail_rank_filter < 10:
                tail_hits10_filter += 1
        print('-----Raw-----')
        head_meanrank_raw /= n_used_eval_triple
        head_hits10_raw /= n_used_eval_triple
        tail_meanrank_raw /= n_used_eval_triple
        tail_hits10_raw /= n_used_eval_triple
        print('-----Head prediction-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(head_meanrank_raw, head_hits10_raw))
        print('-----Tail prediction-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(tail_meanrank_raw, tail_hits10_raw))
        print('------Average------')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format((head_meanrank_raw + tail_meanrank_raw) / 2,
                                                         (head_hits10_raw + tail_hits10_raw) / 2))
        head_meanrank_filter /= n_used_eval_triple
        head_hits10_filter /= n_used_eval_triple
        tail_meanrank_filter /= n_used_eval_triple
        tail_hits10_filter /= n_used_eval_triple
        print('-----Finish evaluation-----')
        raw_res_dict = {'new_triple': n_new_triple, 'used_eval_triple': n_used_eval_triple,
                        'H_MR': head_meanrank_raw, 'H_h10': head_hits10_raw,
                        'T_MR': tail_meanrank_raw, 'T_h10': tail_hits10_raw,
                        'AVE_MR': (head_meanrank_raw + tail_meanrank_raw) / 2,
                        'AVE_h10': (head_hits10_raw + tail_hits10_raw) / 2}
        filter_res_dict = {'new_triple': n_new_triple, 'used_eval_triple': n_used_eval_triple,
                           'H_MR': head_meanrank_filter, 'H_h10': head_hits10_filter,
                           'T_MR': tail_meanrank_filter, 'T_h10': tail_hits10_filter,
                           'AVE_MR': (head_meanrank_filter + tail_meanrank_filter) / 2,
                           'AVE_h10': (head_hits10_filter + tail_hits10_filter) / 2}
        self.kg.result2txt(self.cmp_name, self.cmp_name, raw_res_dict, filter_res_dict)


kg = KnowledgeGraph(data_path='data/WN18/', name='my', seed=1)
kge_model = TransUpdate(cmp_name='transr', kg=kg, dissimilarity_func='L2', learning_rate=0.01, epoch=100, eval_times=5)
gpu_config = tf.GPUOptions(allow_growth=False)
sess_config = tf.ConfigProto(gpu_options=gpu_config)

with tf.Session(config=sess_config) as sess:
    tf.global_variables_initializer().run()
    kge_model.launch_training(sess)
