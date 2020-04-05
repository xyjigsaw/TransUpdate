# Name: gat_layer
# Author: Reacubeth
# Time: 2020/3/27 10:49
# Mail: noverfitting@gmail.com
# Site: www.omegaxyz.com
# *_*coding:utf-8 *_*

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers


class GraphAttentionLayer(keras.layers.Layer):
    def compute_output_signature(self, input_signature):
        pass

    def __init__(self,
                 input_dim,
                 output_dim,
                 adj,
                 nodes_num,
                 dropout_rate=0.0,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 coef_dropout=0.0,
                 **kwargs):
        """
        :param input_dim: 输入的维度
        :param output_dim: 输出的维度，不等于input_dim
        :param adj: 具有自环的tuple类型的邻接表coords, values, shape
        :param nodes_num: 点数量
        :param dropout_rate: 丢弃率，防过拟合，默认0.5
        :param activation: 激活函数
        :param use_bias: 偏移，默认True
        :param kernel_initializer: 权值初始化方法
        :param bias_initializer: 偏移初始化方法
        :param kernel_regularizer: 权值正则化
        :param bias_regularizer: 偏移正则化
        :param activity_regularizer: 输出正则化
        :param kernel_constraint: 权值约束
        :param bias_constraint: 偏移约束
        :param coef_dropout: 互相关系数丢弃，默认0.0
        :param kwargs:
        """
        super(GraphAttentionLayer, self).__init__()
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.support = [tf.SparseTensor(indices=adj[0][0], values=adj[0][1], dense_shape=adj[0][2])]
        self.dropout_rate = dropout_rate
        self.coef_drop = coef_dropout
        self.nodes_num = nodes_num
        self.kernel = None
        self.mapping = None
        self.bias = None

    def build(self, input_shape):
        """
        只执行一次
        """
        self.kernel = self.add_weight(shape=(self.input_dim, self.output_dim),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.nodes_num, self.output_dim),
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        trainable=True)
        print('[GAT LAYER]: GAT W & b built.')

    def call(self, inputs, training=True):
        # 完成输入到输出的映射关系
        # inputs = tf.nn.l2_normalize(inputs, 1)
        raw_shape = inputs.shape
        inputs = tf.reshape(inputs, shape=(1, raw_shape[0], raw_shape[1]))  # (1, nodes_num, input_dim)
        mapped_inputs = keras.layers.Conv1D(self.output_dim, 1, use_bias=False)(inputs)  # (1, nodes_num, output_dim)
        # mapped_inputs = tf.nn.l2_normalize(mapped_inputs)

        sa_1 = keras.layers.Conv1D(1, 1)(mapped_inputs)  # (1, nodes_num, 1)
        sa_2 = keras.layers.Conv1D(1, 1)(mapped_inputs)  # (1, nodes_num, 1)

        con_sa_1 = tf.reshape(sa_1, shape=(raw_shape[0], 1))  # (nodes_num, 1)
        con_sa_2 = tf.reshape(sa_2, shape=(raw_shape[0], 1))  # (nodes_num, 1)

        con_sa_1 = tf.cast(self.support[0], dtype=tf.float32) * con_sa_1  # (nodes_num, nodes_num) W_hi
        con_sa_2 = tf.cast(self.support[0], dtype=tf.float32) * tf.transpose(con_sa_2, [1, 0])  # (nodes_num, nodes_num) W_hj

        weights = tf.sparse.add(con_sa_1, con_sa_2)  # concatenation
        weights_act = tf.SparseTensor(indices=weights.indices,
                                      values=tf.nn.leaky_relu(weights.values),
                                      dense_shape=weights.dense_shape)  # 注意力互相关系数
        attention = tf.sparse.softmax(weights_act)  # 输出注意力机制
        inputs = tf.reshape(inputs, shape=raw_shape)
        if self.coef_drop > 0.0:
            attention = tf.SparseTensor(indices=attention.indices,
                                        values=tf.nn.dropout(attention.values, self.coef_dropout),
                                        dense_shape=attention.dense_shape)
        if training and self.dropout_rate > 0.0:
            inputs = tf.nn.dropout(inputs, self.dropout_rate)
        if not training:
            print("[GAT LAYER]: GAT not training now.")

        attention = tf.sparse.reshape(attention, shape=[self.nodes_num, self.nodes_num])
        value = tf.matmul(inputs, self.kernel)
        value = tf.sparse.sparse_dense_matmul(attention, value)

        if self.use_bias:
            ret = tf.add(value, self.bias)
        else:
            ret = tf.reshape(value, (raw_shape[0], self.output_dim))
        return self.activation(ret)
