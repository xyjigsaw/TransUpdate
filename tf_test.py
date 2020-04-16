# Name: tf_test
# Author: Reacubeth
# Time: 2020/4/3 16:38
# Mail: noverfitting@gmail.com
# Site: www.omegaxyz.com
# *_*coding:utf-8 *_*

import tensorflow.compat.v1 as tf
import numpy as np
tf.compat.v1.disable_eager_execution()

a = tf.constant([1, 2, 4, 5, 6], dtype=tf.int32)
print(a)

b = tf.constant([2, 7, 1, 5, 4, 7], dtype=tf.int32)

para1 = tf.shape(a)[0]
para2 = tf.shape(b)[0]
sum_para = tf.add(para1, para2)
para1 = tf.divide(para1, sum_para)
para2 = tf.divide(para2, sum_para)
tilex = tf.tile(a, [5, 0])

with tf.Session() as sess:
    print(sess.run(tilex))
