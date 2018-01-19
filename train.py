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

trainn, testn, train_label, test_label=loaddata(dataset)
print(trainn)
save_as_tfrecord(trainn, train_label, "./dataset/train.bin")
save_as_tfrecord(testn, test_label, "./dataset/test.bin")
m=np.shape(trainn)[0]
IMG_SIZE = 128  # 图像大小

X = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 3])
Y = tf.placeholder(tf.float32,[None, ])

#init weights
weights = {
    "w1":init_weights([3,3,3,16]),
    "w2":init_weights([3,3,16,32]),
    "w3":init_weights([3,3,32,64]),
    "w4":init_weights([3,3,64,100]),
    "w5":init_weights([3,3,100,128]),
    "w6":init_weights([2048,256]),
    "w7":init_weights([256,64]),
    "wo":init_weights([64, 1])
    }

#init biases
biases = {
    "b1":init_weights([16]),
    "b2":init_weights([32]),
    "b3":init_weights([64]),
    "b4":init_weights([100]),
    "b5":init_weights([128]),
    "b6":init_weights([256]),
    "b7":init_weights([64]),
    "bo":init_weights([1])
    }
def model(X, weights, biases):
    l1a = conv2d(X,weights["w1"],biases["b1"])
    l1 = pooling(l1a)

    l2a = conv2d(l1,weights["w2"],biases["b2"])
    l2 = pooling(l2a)

    l3a = conv2d(l2,weights["w3"],biases["b3"])
    l3 = pooling(l3a)

    l4a = conv2d(l3,weights["w4"],biases["b4"])
    l4 = pooling(l4a)

    l5a = conv2d(l4,weights["w5"],biases["b5"])
    l5 = pooling(l5a)

    l6a = tf.reshape(l5,[-1,weights["w6"].get_shape().as_list()[0]])
    l6 = tf.nn.relu(tf.matmul(l6a,weights["w6"])+biases["b6"])

    l7a = tf.reshape(l6,[-1,weights["w7"].get_shape().as_list()[0]])
    l7 = tf.nn.relu(tf.matmul(l7a,weights["w7"])+biases["b7"])

    y_pred = tf.add(tf.matmul(l7, weights["wo"]), biases["bo"])

    return y_pred
# y_pred是预测tensor
y_pred = model(X, weights, biases)

# 定义损失函数

cost=tf.reduce_sum(tf.square(y_pred - Y))

train_op = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam').minimize(cost)

# 获取batch。注意这里是tensor，需要运行
img_batch, label_batch = input_pipeline(["./dataset/train.bin"], 1)

iteration = 200
disp_step = 100
idisp_step=1000
save_step = 100
max_step = 200
cost_history=[]
step = 0
coststep = []
saver = tf.train.Saver()
yaxis=0

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        # 获取训练数据成功，并且没有到达最大训练次数
        while not coord.should_stop() and step < max_step:
            step += 1
            # 运行tensor，获取数据
            for i in range(m):
                imgs, labels = sess.run([img_batch, label_batch])
                sess.run(train_op, feed_dict={X: imgs, Y: labels})
                cost_history.append(sess.run(cost, feed_dict={X: imgs,Y: labels}))
                yaxis=yaxis+1
                coststep.append(yaxis)
                if i % idisp_step == 0:
                    cos = sess.run(cost, feed_dict={X: imgs,Y: labels})
                    print('Epoch:',step ,'Sample %s cost is %.2f' % (i, cos))
                # if step % disp_step == 0:
                #     cos = sess.run(cost, feed_dict={X: imgs,Y: labels})
                #     print('%s cost is %.2f' % (step, cos))
            if step % save_step == 0:
                # 保存当前模型
                save_path = saver.save(sess, './model/wejump.ckpt', global_step=step)
                print("save graph to %s" % save_path)
    except tf.errors.OutOfRangeError:
        print("reach epoch limit")
    finally:
        coord.request_stop()
    coord.join(threads)
    save_path = saver.save(sess, './model/wejump.ckpt', global_step=step)

plt.plot(coststep, cost_history, label="Cost")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost Graph")
plt.legend()
plt.show()




