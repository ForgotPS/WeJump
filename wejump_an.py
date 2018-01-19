#encoding=utf-8

# coding: utf-8
import os
import sys
import subprocess
import time
import random
from PIL import Image

import tensorflow as tf
from tensorflow.python.framework import ops
from common import screenshot, debug
from cnn_utils import *

SCALE = 1.02


def set_button_position(im):
    global swipe_x1, swipe_y1, swipe_x2, swipe_y2
    w, h = im.size
    print(w,h)
    left = int(w / 2)
    top = int(1584 * (h / 1920.0))
    left = int(random.uniform(left-50, left+50))
    top = int(random.uniform(top-10, top+10))
    swipe_x1, swipe_y1, swipe_x2, swipe_y2 = left, top, left, top

def jump(press_time, SCALE):
    press_time = int(press_time*SCALE)
    cmd = 'adb shell input swipe {x1} {y1} {x2} {y2} {duration}'.format(
        x1=swipe_x1,
        y1=swipe_y1,
        x2=swipe_x2,
        y2=swipe_y2,
        duration=press_time
    )
    os.system(cmd)


def main():
    debug.dump_device_info()
    screenshot.check_screenshot()
    label=[0]
    while True:
        screenshot.pull_screenshot()
        image = Image.open('./temp.png')
        set_button_position(image)
        tempimage=['.\\temp.png']
        save_tfrecord(tempimage, label, "./temp/temp.bin")
        img_batch, label_batch = input_pipeline(["./temp/temp.bin"], 1)
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            saver = tf.train.Saver()
            saver.restore(sess, ("./model/wejump.ckpt-2000"))
            W = sess.run(weights)
            B = sess.run(biases)
            imgs, labels = sess.run([img_batch, label_batch])
            press_time = tf.Session().run(modeltest(imgs,W,B))[0][0]
            print('Presstime:',press_time)
            jump(press_time, SCALE)
        time.sleep(random.uniform(2, 2.5))

if __name__ == '__main__':
    main()
