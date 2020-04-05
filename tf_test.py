# Name: tf_test
# Author: Reacubeth
# Time: 2020/4/3 16:38
# Mail: noverfitting@gmail.com
# Site: www.omegaxyz.com
# *_*coding:utf-8 *_*

import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
X = tf.constant([[2.0, 2.0, 3.0]])
Z = tf.constant([[2.0, 4.0, 5.0]])

y_2 = tf.reduce_mean(X, 0)  # 沿着轴0求和
y_3 = tf.reduce_sum(X, 1, keep_dims=True)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    c_1 = sess.run(-tf.reduce_sum(tf.log(X)*Z, axis=1))
    print(c_1)
    # print(c_2)
    # print(c_3)
