#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


import tensorflow as tf 
import numpy as np

from read_data_cpm import read_and_decode

joints_num = 16
batch_size = 1

def conv2d(inputs, shape, strides, padding='SAME', stddev=0.005, activation_fn=True):

    kernel = tf.Variable(tf.truncated_normal(shape,
                                            dtype=tf.float32, stddev=stddev))    
    conv = tf.nn.conv2d(inputs, kernel, strides, padding=padding)
    bias = tf.Variable(tf.truncated_normal([shape[-1]],
                                            dtype=tf.float32, stddev=stddev))
    conv = tf.nn.bias_add(conv, bias)
    if activation_fn:
        conv = tf.nn.relu(conv)
    return conv

def muti_conv2d(inputs):

    Mconv1_stage = conv2d(inputs, [11, 11, 49, 128], [1, 1, 1, 1])
    Mconv2_stage = conv2d(Mconv1_stage, [11, 11, 128, 128], [1, 1, 1, 1])
    Mconv3_stage = conv2d(Mconv2_stage, [11, 11, 128, 128], [1, 1, 1, 1])
    Mconv4_stage = conv2d(Mconv3_stage, [1, 1, 128, 128], [1, 1, 1, 1])
    Mconv5_stage = conv2d(Mconv4_stage, [1, 1, 128, joints_num], [1, 1, 1, 1], activation_fn=False)

    return Mconv5_stage

def cpm(images, center_map):

    pool_center_map = tf.nn.avg_pool(center_map, ksize=[1, 9, 9, 1], strides=[1, 8, 8, 1], padding='SAME')


    #stage 1
    conv1_stage1 = conv2d(images, [9, 9, 3, 128], [1, 1, 1, 1])
    pool1_stage1 = tf.nn.max_pool(conv1_stage1, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')
    
    conv2_stage1 = conv2d(pool1_stage1, [9, 9, 128, 128], [1, 1, 1, 1])
    pool2_stage1 = tf.nn.max_pool(conv2_stage1, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')

    conv3_stage1 = conv2d(pool2_stage1, [9, 9, 128, 128], [1, 1, 1, 1])
    pool3_stage1 = tf.nn.max_pool(conv3_stage1, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')

    conv4_stage1 = conv2d(pool3_stage1, [5, 5, 128, 32], [1, 1, 1, 1])

    conv5_stage1 = conv2d(conv4_stage1, [9, 9, 32, 512], [1, 1, 1, 1])

    conv6_stage1 = conv2d(conv5_stage1, [1, 1, 512, 512], [1, 1, 1, 1])

    conv7_stage1 = conv2d(conv6_stage1, [1, 1, 512, joints_num], [1, 1, 1, 1], activation_fn=False)
    tf.add_to_collection('heatmaps', conv7_stage1)
    
    #stage 2
    conv1_stage2 = conv2d(images, [9, 9, 3, 128], [1, 1, 1, 1])
    pool1_stage2 = tf.nn.max_pool(conv1_stage2, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')
    
    conv2_stage2 = conv2d(pool1_stage2, [9, 9, 128, 128], [1, 1, 1, 1])
    pool2_stage2 = tf.nn.max_pool(conv2_stage2, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')

    conv3_stage2 = conv2d(pool2_stage2, [9, 9, 128, 128], [1, 1, 1, 1])
    pool3_stage2 = tf.nn.max_pool(conv3_stage2, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')

    conv4_stage2 = conv2d(pool3_stage2, [5, 5, 128, 32], [1, 1, 1, 1])    

    concat_stage2 = tf.concat(axis=3, values=[conv4_stage2, conv7_stage1, pool_center_map])

    Mconv_stage2 = muti_conv2d(concat_stage2)
    tf.add_to_collection('heatmaps', Mconv_stage2)

    #stage3
    conv1_stage3 = conv2d(pool3_stage2, [5, 5, 128, 32], [1, 1, 1, 1])
    concat_stage3 = tf.concat(axis=3, values=[conv1_stage3, Mconv_stage2, pool_center_map])
    Mconv_stage3 = muti_conv2d(concat_stage3)
    tf.add_to_collection('heatmaps', Mconv_stage3)

    #stage4
    conv1_stage4 = conv2d(pool3_stage2, [5, 5, 128, 32], [1, 1, 1, 1])
    concat_stage4 = tf.concat(axis=3, values=[conv1_stage4, Mconv_stage3, pool_center_map])
    Mconv_stage4 = muti_conv2d(concat_stage4)
    tf.add_to_collection('heatmaps', Mconv_stage4)

    # stage5
    conv1_stage5 = conv2d(pool3_stage2, [5, 5, 128, 32], [1, 1, 1, 1])
    concat_stage5 = tf.concat(axis=3, values=[conv1_stage5, Mconv_stage4, pool_center_map])
    Mconv_stage5 = muti_conv2d(concat_stage5)
    tf.add_to_collection('heatmaps', Mconv_stage5)

    # stage6
    conv1_stage6 = conv2d(pool3_stage2, [5, 5, 128, 32], [1, 1, 1, 1])
    concat_stage6 = tf.concat(axis=3, values=[conv1_stage6, Mconv_stage5, pool_center_map])
    Mconv_stage6 = muti_conv2d(concat_stage6)


    return Mconv_stage6

def train():
        
    image_batch, heatmap_batch, center_map_batch = read_and_decode(['./{}.tfrecords'.format(i) for i in xrange(23)], batch_size)
   

    output = cpm(image_batch, center_map_batch)
    loss = tf.nn.l2_loss(heatmap_batch - output)
    interm_loss = tf.reduce_sum(tf.stack([tf.nn.l2_loss(heatmap_batch - o) for o in tf.get_collection('heatmaps')]))
    total_loss = loss + interm_loss


    optimizer = tf.train.AdamOptimizer(1e-4).minimize(total_loss)

    with tf.Session() as sess:

        saver = tf.train.Saver()  
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()  
        threads = tf.train.start_queue_runners(coord=coord)

        iter_time = time.time()
        
        for iters in xrange(1000000):
            
            _, e = sess.run([optimizer, loss])

            if iters % 100 == 0:
                print iters, e, time.time() - iter_time
                iter_time = time.time()
            if iters % 200000 == 0:
                save_path = saver.save(sess, './model_%d.ckpt'%iters) 

        coord.request_stop()  
        coord.join(threads)

        save_path = saver.save(sess, 'models/model_final.ckpt') 
        sess.close() 
       

def train_single():
    from view import view

    
    image_batch, heatmap_batch, center_map_batch = read_and_decode(['{}.tfrecords'.format(i) for i in xrange(3)], batch_size)
   
    inputs = tf.placeholder(tf.float32, shape = [batch_size, None, None, 3])
    gt_heatmap = tf.placeholder(tf.float32, shape = [batch_size, None, None, joints_num])
    center_map = tf.placeholder(tf.float32, shape = [batch_size, None, None, 1])
    output = cpm(inputs, center_map)
    loss = tf.nn.l2_loss(gt_heatmap - output)
    interm_loss = tf.reduce_sum(tf.stack([tf.nn.l2_loss(gt_heatmap - o) for o in tf.get_collection('heatmaps')]))

    total_loss = loss + interm_loss

    optimizer = tf.train.AdamOptimizer(1e-4).minimize(total_loss)

    with tf.Session() as sess:

        saver = tf.train.Saver()  
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()  
        threads = tf.train.start_queue_runners(coord=coord)

        iter_time = time.time()
        
        images, heatmaps, c_map= sess.run([image_batch, heatmap_batch, center_map_batch])

        view(images[0,:,:,:], heatmaps[0,:,:,:], show_max=False)
        for iters in xrange(150):
            
            _, e = sess.run([optimizer, loss], feed_dict={inputs:images, center_map:c_map, gt_heatmap:heatmaps})
            print iters, e

        pred = sess.run(output, feed_dict={inputs:images, center_map:c_map})
        view(images[0,:,:,:], pred[0,:,:,:], show_max=True)
        view(images[0,:,:,:], pred[0,:,:,:], show_max=False)
        coord.request_stop()  
        coord.join(threads)
        sess.close() 


if __name__ == '__main__':

    train_single()