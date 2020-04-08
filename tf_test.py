# Name: tf_test
# Author: Reacubeth
# Time: 2020/4/3 16:38
# Mail: noverfitting@gmail.com
# Site: www.omegaxyz.com
# *_*coding:utf-8 *_*

import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

a = tf.constant([[1, 2, 4, 5, 6]], dtype=tf.float32)
print(a)

a1 = tf.tile(a, [2, 1])

with tf.Session() as sess:
    print(sess.run(a1))
