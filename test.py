#encoding=utf-8
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from cnn_utils import *


dataset = "./dataset/images"
yaxis = 0
cost_history,coststep=[],[]
IMG_SIZE = 128
print('Placeholder')
X = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 3])
Y = tf.placeholder(tf.float32,[None, ])
print('Import Test.bin')
img_batch, label_batch = input_pipeline(["./dataset/test.bin"], 1)
print('Import Completed.')
trainn, testn, train_label, test_label=loaddata(dataset)
m=np.shape(testn)[0]
print('Total Sample Number:',m)
print('Start Sess')
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    print('Load Model')
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint("./model"))
    W = sess.run(weights)
    B = sess.run(biases)
    print('Model load successfully.')
    for i in range(m):
        imgs, labels = sess.run([img_batch, label_batch])
        y_pred = modeltest(imgs,W,B)
        cost = tf.reduce_sum(tf.square(y_pred - Y))
        testcost = sess.run(cost, feed_dict={X: imgs, Y: labels})
        cost_history.append(testcost)
        yaxis = yaxis + 1
        coststep.append(yaxis)


plt.plot(coststep, cost_history, label="Cost")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost Graph")
plt.legend()
plt.show()

    # print(sess.run(weights))
    # print(sess.run(biases))
# cost = tf.reduce_sum(tf.square(y_pred - Y))
# with tf.Session() as sess:
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     for i in range(m):
#         sess.run(cost, feed_dict={X: imgs, Y: labels})
#         print(i,':',cost)
#


