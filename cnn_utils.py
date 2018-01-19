import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import json
import os
from PIL import Image
import random

def inm_return_presstime(imagename):
    train_presstime = json.load(open("./dataset/dataset.json"))
    presstime = train_presstime[imagename]
    return presstime

def fulltofile(fullnames):
    filename=fullnames.split("\\")[-1]
    return filename

def loaddata(dataset):

    fullnames = [os.path.join(dataset, image_file) \
             for image_file in os.listdir(dataset)]
    m=np.shape(fullnames)[0]
    filenames=[]
    for i in range(m):
        filenames.append(fulltofile(fullnames[i]))
    filenumbers= len(filenames)
    np.random.shuffle(np.arange(len(filenames)))
    TRAIN_SEC, TEST_SEC = 0.9, 0.1
    trainn, testn = fullnames[: int(filenumbers * TRAIN_SEC)], fullnames[int(filenumbers * TRAIN_SEC) :]
    # print("Sample : train num is %d, test num is %d" % (len(trainn), len(testn)))
    train_label,test_label=[],[]
    for trn in range(len(trainn)):
        train_label.append(inm_return_presstime(fulltofile(trainn[trn])))
    for tst in range(len(testn)):
        test_label.append(inm_return_presstime(fulltofile(testn[tst])))

    return trainn, testn, train_label, test_label

# print(train_label)
# print(test_label)
# print(trainn[0])
# print(testn[0])
def set_button_position(im):
    global swipe_x1, swipe_y1, swipe_x2, swipe_y2
    w, h = im.size
    print(w,h)
    left = int(w / 2)
    top = int(1584 * (h / 1920.0))
    left = int(random.uniform(left-50, left+50))
    top = int(random.uniform(top-10, top+10))
    swipe_x1, swipe_y1, swipe_x2, swipe_y2 = left, top, left, top


def resize_img(img_path, shape):
    '''
        resize image given by `image_path` to `shape`
    '''
    im = Image.open(img_path)
    im = im.resize(shape)
    im = im.convert('RGB')
    return im



def save_as_tfrecord(samples, labels, bin_path):
    '''
        Save images and labels as TFRecord to file: `bin_path`
    '''
    assert len(samples) == len(labels)
    writer = tf.python_io.TFRecordWriter(bin_path)
    img_label = list(zip(samples, labels))
    np.random.shuffle(img_label)
    for img, label in img_label:
        im = resize_img(img, (128, 128))
        im_raw = im.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[im_raw]))
        }))
        writer.write(example.SerializeToString())
    writer.close()

def resize_imgtf(img_path, shape):
    '''
        resize image given by `image_path` to `shape`
    '''
    image = Image.open(img_path)
    w, h = image.size
    top = (h - w) / 2
    im = image.crop((0, top, w, w + top))
    im = im.convert('RGB')
    im = im.resize(shape)
    return im

def save_tfrecord(samples, labels, bin_path):
    '''
        Save images and labels as TFRecord to file: `bin_path`
    '''
    assert len(samples) == len(labels)
    writer = tf.python_io.TFRecordWriter(bin_path)
    img_label = list(zip(samples, labels))
    np.random.shuffle(img_label)
    for img, label in img_label:
        im = resize_imgtf(img, (128, 128))
        im_raw = im.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[im_raw]))
        }))
        writer.write(example.SerializeToString())
    writer.close()

def loaddatatf(dataset):
    fullnames = [os.path.join(dataset, image_file) \
                for image_file in os.listdir(dataset)]
    fullnamestf = tf.constant(fullnames)
    m=np.shape(fullnames)[0]
    filenames=[]
    for i in range(m):
        filenames.append(fulltofile(fullnames[i]))
    filenumbers= len(filenames)
    np.random.shuffle(np.arange(len(filenames)))
    TRAIN_SEC, TEST_SEC = 0.9, 0.1
    trainn, testn = fullnames[: int(filenumbers * TRAIN_SEC)], fullnames[int(filenumbers * TRAIN_SEC) :]
    # print("Sample : train num is %d, test num is %d" % (len(trainn), len(testn)))
    train_label,test_label=[],[]
    for trn in range(len(trainn)):
        train_label.append(inm_return_presstime(fulltofile(trainn[trn])))
    for tst in range(len(testn)):
        test_label.append(inm_return_presstime(fulltofile(testn[tst])))

    trainntf=tf.constant(trainn)
    testntf = tf.constant(testn)
    train_labeltf, test_labeltf = tf.constant(train_label), tf.constant(test_label),
    return fullnames,trainn, testn, train_label, test_label, fullnamestf,trainntf, testntf, train_labeltf, test_labeltf

def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_image(image_string)
  image_resized = tf.image.resize_images(image_decoded, [128, 128])
  return image_resized, label




# 获取并初始化权重
def init_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def conv2d(x,w,b):
    x = tf.nn.conv2d(x,w,strides = [1,1,1,1],padding = "SAME")
    x = tf.nn.bias_add(x,b)
    return tf.nn.relu(x)

def pooling(x):
    return tf.nn.max_pool(x,ksize = [1,2,2,1],strides = [1,2,2,1],padding = "SAME")


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

def modeltest(X, weights, biases):
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

    l6a = tf.reshape(l5,[-1,np.shape(weights["w6"])[0]])
    l6 = tf.nn.relu(tf.matmul(l6a,weights["w6"])+biases["b6"])

    l7a = tf.reshape(l6,[-1,np.shape(weights["w7"])[0]])
    l7 = tf.nn.relu(tf.matmul(l7a,weights["w7"])+biases["b7"])

    y_pred = tf.add(tf.matmul(l7, weights["wo"]), biases["bo"])

    return y_pred

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #return filename and example
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [128, 128, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5 # normalize
    label = tf.cast(features['label'], tf.int32)
    #label = tf.sparse_to_dense(label, [1], 10000, 0)
    return img, label

def input_pipeline(filenames, batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)
    example, label = read_and_decode(filename_queue)
    min_after_dequeue = 1
    num_threads = 2
    capacity = min_after_dequeue + (num_threads + 1) * batch_size
    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size=batch_size, capacity=capacity, num_threads = num_threads,
        min_after_dequeue=min_after_dequeue)
    return example_batch, label_batch



