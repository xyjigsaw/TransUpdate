# Name: transUpdate
# Author: Reacubeth
# Time: 2020/3/26 14:16
# Mail: noverfitting@gmail.com
# Site: www.omegaxyz.com
# *_*coding:utf-8 *_*

import timeit
import tensorflow.compat.v1 as tf
from KG_data import KnowledgeGraph
from gat_layer import GraphAttentionLayer
from layout import sparse_fruchterman_reingold
import numpy as np

tf.compat.v1.disable_eager_execution()


class TransUpdate:
    def __init__(self, kg: KnowledgeGraph, dissimilarity_func, learning_rate, epoch):
        # 根据文献 lr = {0.001, 0.01, 0.1} embedding_dim = {20, 50}
        # WN18数据集 lr = 0.01  dim = 50  d_f = L1
        # FB15K lr = 0.01  dim = 50  d_f = L2
        self.epoch = epoch

        self.kg = kg  # 知识图谱三元组
        self.dissimilarity_func = dissimilarity_func  # 不相似函数(稀疏函数) 一般取 L1 或 L2
        self.learning_rate = learning_rate  # 学习率
        # 初始化一个三元组
        self.triple = tf.placeholder(dtype=tf.int32, shape=[None, 3])
        self.sparse_neighbor = tf.placeholder(dtype=tf.int32)
        self.sparse_neighbor_noself = tf.placeholder(dtype=tf.int32)
        self.prob_e = tf.placeholder(dtype=tf.float32)

        self.train_op = None  # 训练操作，是最小化优化器的那一步 如 tf.train.AdamOptimizer().minimize()
        self.loss = None  # 损失函数
        self.global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')  # 全局训练步数

        # 初始embedding
        self.entity_embedding = None
        self.relation_embedding = None
        self.embedding_dim = None
        self.load_embedding()  # 读取初始词嵌入和维度

        self.init_train()  # 初始化训练步骤

        # 评估操作
        self.eval_triple = tf.placeholder(dtype=tf.int32, shape=[3])  # 评估三元组，3行n列
        self.idx_head_prediction = None
        self.idx_tail_prediction = None
        self.build_eval_graph()

        # 评估操作
        self.eval_triple4raw = tf.placeholder(dtype=tf.int32, shape=[3])  # 评估三元组，3行n列
        self.idx_head_prediction4raw = None
        self.idx_tail_prediction4raw = None
        self.entity_embedding4raw = None
        self.relation_embedding4raw = None
        self.load_embedding4raw()
        self.build_eval_graph4raw()

    def load_embedding(self):
        """
        读取已有嵌入模型
        """
        start = timeit.default_timer()
        f = open("output/entity_embedding.txt", "r")
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

        f = open("output/relation_embedding.txt", "r")
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
        print('EMBEDDING LOADING COST TIME: {:.4f}s.'.format(timeit.default_timer() - start))

    def init_train(self):
        # 正向传播 计算损失 训练步骤
        # 正则化
        self.entity_embedding = tf.nn.l2_normalize(self.entity_embedding, dim=1)
        self.relation_embedding = tf.nn.l2_normalize(self.relation_embedding, dim=1)

        p_neighbor = self.get_prob(self.triple, self.sparse_neighbor)
        p_real_neighbor = self.get_real_prob(self.sparse_neighbor_noself)

        # distance = p_neighbor - p_real_neighbor
        self.loss = tf.reduce_mean(-tf.reduce_sum(tf.log(p_neighbor)*p_real_neighbor, axis=1))

        '''
        # 计算损失 与选择的dissimilarity_func有关
        if self.dissimilarity_func == 'L1':
            # 将每一行相加 所有pos/neg的得分
            score = tf.reduce_sum(tf.abs(distance), axis=1)
        else:  # L2 score
            score = tf.reduce_sum(tf.square(distance), axis=1)

        # 使用relu函数取正数部分
        self.loss = tf.reduce_sum(score)
        # 选择优化器，并继续训练操作
        '''
        self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate). \
            minimize(self.loss, global_step=self.global_step)

    def get_prob(self, triple, sparse_neighbor):
        head = tf.nn.embedding_lookup(self.entity_embedding, triple[:, 0])
        # p_h = tf.Variable(tf.zeros([1, self.kg.n_entity]), name="p_h")
        v_neighbors = tf.nn.embedding_lookup(self.entity_embedding, sparse_neighbor)
        sum_neighbor_mat = tf.exp(tf.matmul(v_neighbors, tf.transpose(head)))
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
            if n_used_triple % 5 == 0:
                try:
                    self.launch_evaluation(sess, n_used_triple)
                    self.launch_evaluation4raw(sess, n_used_triple)
                except ZeroDivisionError as e:
                    print('ERROR:', n_used_triple)
            n_used_triple += 1
            print("=" * 20 + "TRAINING FACT {}".format(n_used_triple) + "=" * 20)
            print('triple: ', triple)
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
            # print('prob_e', prob_e)
            # print('rel_cnt', rel_cnt)

            epoch_loss = 0
            for i in range(self.epoch):
                regret_loss, _ = sess.run(fetches=[self.loss, self.train_op],
                                          feed_dict={self.triple: triple,
                                                     self.sparse_neighbor: np.array(list(neighbor_id_rel.keys())),
                                                     self.sparse_neighbor_noself: np.array(list(neighbor_id_rel.keys())[:-1]),
                                                     self.prob_e: prob_e})
                # print('regret_loss', regret_loss)
                epoch_loss += regret_loss
                print('REGRET: {:.5f}'.format(regret_loss))
            print('[TRAIN {:.3f}s] #triple: {}/{} Done.'.format(timeit.default_timer() - start,
                                                                n_used_triple,
                                                                self.kg.n_train_triple))
            print('EPOCH LOSS (all regret single fact): {:.3f}'.format(epoch_loss))
            print('EMB:', np.array(self.entity_embedding.eval()).sum())
            self.kg.sparse_one_adj = None
            self.kg.sparse_two_adj = None
            self.kg.no_weighted_adj(self.kg.n_entity, self.kg.fact_triple)

    def my_gat_layer(self, input_embeds, num, in_dim, out_dim):
        # self.entity_embedding = self.my_gat_layer(self.entity_embedding, self.kg.n_entity, self.embedding_dim, self.embedding_dim)
        g_layer = GraphAttentionLayer(input_dim=in_dim,
                                      output_dim=out_dim,
                                      adj=[self.kg.one_adj],
                                      nodes_num=num,
                                      num_features_nonzero=0,
                                      alpha=0.0,
                                      activations='tanh',
                                      dropout_rate=0.5)

        return g_layer(input_embeds)

    def build_eval_graph(self):
        with tf.name_scope('evaluation'):
            self.idx_head_prediction, self.idx_tail_prediction = self.evaluate(self.eval_triple)

    def evaluate(self, eval_triple):
        with tf.name_scope('lookup'):
            head = tf.nn.embedding_lookup(self.entity_embedding, eval_triple[0])  # 头结点
            tail = tf.nn.embedding_lookup(self.entity_embedding, eval_triple[2])  # 尾结点
            relation = tf.nn.embedding_lookup(self.relation_embedding, eval_triple[1])  # 关系

        with tf.name_scope('link'):
            # 头结点预测 通过训练的节点嵌入来代替原始头结点
            distance_head_prediction = self.entity_embedding + relation - tail

            # 尾结点预测 通过训练的节点嵌入来代替原始尾结点
            distance_tail_prediction = head + relation - self.entity_embedding

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
                                                                   feed_dict={self.eval_triple: eval_triple})
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
        #print('-----Filter-----')
        head_meanrank_filter /= n_used_eval_triple
        head_hits10_filter /= n_used_eval_triple
        tail_meanrank_filter /= n_used_eval_triple
        tail_hits10_filter /= n_used_eval_triple
        #print('-----Head prediction-----')
        #print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(head_meanrank_filter, head_hits10_filter))
        #print('-----Tail prediction-----')
        #print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(tail_meanrank_filter, tail_hits10_filter))
        #print('-----Average-----')
        #print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format((head_meanrank_filter + tail_meanrank_filter) / 2,
        #                                                 (head_hits10_filter + tail_hits10_filter) / 2))
        #print('cost time: {:.3f}s'.format(timeit.default_timer() - start))
        #print('-----Finish evaluation-----')
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
        self.kg.result2txt('my', raw_res_dict, filter_res_dict)


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
    def load_embedding4raw(self):
        """
        读取已有嵌入模型
        """
        f = open("output/entity_embedding.txt", "r")
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

        f = open("output/relation_embedding.txt", "r")
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

    def build_eval_graph4raw(self):
        with tf.name_scope('evaluation4raw'):
            self.idx_head_prediction4raw, self.idx_tail_prediction4raw = self.evaluate4raw(self.eval_triple4raw)

    def evaluate4raw(self, eval_triple4raw):
        with tf.name_scope('lookup4raw'):
            head = tf.nn.embedding_lookup(self.entity_embedding4raw, eval_triple4raw[0])  # 头结点
            tail = tf.nn.embedding_lookup(self.entity_embedding4raw, eval_triple4raw[2])  # 尾结点
            relation = tf.nn.embedding_lookup(self.relation_embedding4raw, eval_triple4raw[1])  # 关系

        with tf.name_scope('link4raw'):
            # 头结点预测 通过训练的节点嵌入来代替原始头结点
            distance_head_prediction = self.entity_embedding4raw + relation - tail

            # 尾结点预测 通过训练的节点嵌入来代替原始尾结点
            distance_tail_prediction = head + relation - self.entity_embedding4raw

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
                                                                   feed_dict={self.eval_triple4raw: eval_triple4raw})
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
        #print('-----Filter-----')
        head_meanrank_filter /= n_used_eval_triple
        head_hits10_filter /= n_used_eval_triple
        tail_meanrank_filter /= n_used_eval_triple
        tail_hits10_filter /= n_used_eval_triple
        #print('-----Head prediction-----')
        #print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(head_meanrank_filter, head_hits10_filter))
        #print('-----Tail prediction-----')
        #print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(tail_meanrank_filter, tail_hits10_filter))
        #print('-----Average-----')
        #print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format((head_meanrank_filter + tail_meanrank_filter) / 2,
        #                                                 (head_hits10_filter + tail_hits10_filter) / 2))
        #print('cost time: {:.3f}s'.format(timeit.default_timer() - start))
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
        self.kg.result2txt('transe', raw_res_dict, filter_res_dict)


kg = KnowledgeGraph(data_path='data/WN18/', name='my')
kge_model = TransUpdate(kg=kg, dissimilarity_func='L2', learning_rate=0.01, epoch=100)
gpu_config = tf.GPUOptions(allow_growth=False)
sess_config = tf.ConfigProto(gpu_options=gpu_config)

with tf.Session(config=sess_config) as sess:
    tf.global_variables_initializer().run()
    kge_model.launch_training(sess)
