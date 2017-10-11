#!/usr/bin/env python
# -*- coding: utf-8 -*-
from multiprocessing import Pool
import numpy as np
import tensorflow as tf
from scipy.misc import imread, imresize



def read_labeled_image_list(file):

    filepaths = []
    labels = []
    sizes = []
    centers = []
    joints_list = []
    with open(file, 'r') as f:
        for line in f.readlines():
            filepath, label = line.split('\t')
            filepaths.append(filepath)
            label = [float(label) for label in ' '.join(label.split()).split(' ')]
            points = zip(*([iter(label)]*2))
            size, center = points[:2]
            joints = points[2:]
            centers.append([center])
            joints_list.append(joints)
            
    return filepaths, centers, joints_list

def gaussian_generator(points, height, width, sigma):

    x = np.arange(0, width, 1, np.float32)
    y = np.arange(0, height, 1,np.float32)[:,np.newaxis]
    channel = len(points)
    heatmap = np.zeros([height, width, channel], dtype=np.float32)

    for c in xrange(channel):
        x0, y0 = points[c]
        if x0 < 0 or y0 < 0:
            continue

        heatmap[:,:,c] = np.exp(-((x -x0) **2 + 
                                (y - y0) ** 2) / 2.0 / sigma / sigma)
    
    return heatmap.tobytes()

def write_tfrecord(examples):
    index = examples.pop()
    tfrecord_name = '{}/{}.tfrecords'.format(path, index)

    with tf.python_io.TFRecordWriter(tfrecord_name) as writer:
        for example in examples:
            filepath, center, joints = example
            
            img = imread(filepath)
            height, width, _ = img.shape  
            if height > 1000:
                height = int(height / 1.5)
                width = int(height / 1.5)

            img = imresize(img, (height, width))
            img_raw = img.tobytes()

            sigma = max(height, width) * 3.0 / 128

            center_map = gaussian_generator(center, height, width, sigma)
            heatmap = gaussian_generator(joints, height, width, sigma)          

            data_example = tf.train.Example(features=tf.train.Features(feature={
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                'center_map': tf.train.Feature(bytes_list=tf.train.BytesList(value=[center_map])),
                'heatmap': tf.train.Feature(bytes_list=tf.train.BytesList(value=[heatmap])),
                'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width]))
                }))
            writer.write(data_example.SerializeToString())

            return

def augment(feature_map, h, w, height, width, random_scale):

    shape = tf.cast([h, w], tf.float32)
    shape = tf.cast(shape * random_scale, tf.int32)
    feature_map = tf.image.resize_images(feature_map, shape)
    feature_map = tf.image.resize_image_with_crop_or_pad(feature_map, height, width)

    return feature_map


def read_and_decode(file_list, batch_size):
    file_name_queue = tf.train.string_input_producer(file_list)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_name_queue)
    features = tf.parse_single_example(serialized_example,
                                        features={
                                        'image': tf.FixedLenFeature([], tf.string),
                                        'center_map': tf.FixedLenFeature([], tf.string),
                                        'heatmap': tf.FixedLenFeature([], tf.string),
                                        'height': tf.FixedLenFeature([], tf.int64),
                                        'width': tf.FixedLenFeature([], tf.int64)
                                        })
    
    image = tf.decode_raw(features['image'], tf.uint8)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)


    image = tf.reshape(image, [height, width, 3])
    image = tf.image.per_image_standardization(image)
    center_map = tf.decode_raw(features['center_map'], tf.float32)
    center_map = tf.reshape(center_map, [height, width, 1])
    heatmap = tf.decode_raw(features['heatmap'], tf.float32)
    heatmap = tf.reshape(heatmap, [height, width, 16])

    random_scale = tf.random_uniform(shape=[1], minval=0.7, maxval=1.3, dtype=tf.float32)
    image = augment(image, height, width, 1080, 1080, random_scale)
    heatmap = augment(heatmap, height, width, 1080, 1080, random_scale)
    center_map = augment(center_map, height, width, 1080, 1080, random_scale)

    heatmap = tf.image.resize_images(heatmap, [1080 / 8, 1080 / 8])

    image, heatmap, center_map = tf.train.shuffle_batch([image, heatmap, center_map], 
                                                                        batch_size=batch_size, 
                                                                        capacity=300, 
                                                                        min_after_dequeue=10)



    return image, heatmap, center_map


def main():
    image_num = 10000
    pool = Pool()

    image_list, center_list, joints_list = read_labeled_image_list('data_cpm.txt')

    example_num = len(image_list)
    example = zip(image_list, center_list, joints_list)
    example_zip = [list(e) for e in zip(*([iter(example)])*image_num)]
    example_zip.append(example[example_num/image_num*image_num:])
    [example_zip[i].append(i) for i in xrange(len(example_zip))]

    pool.map(write_tfrecord, example_zip)
    # [write_tfrecord(example) for example in example_zip]
        


if __name__ == '__main__':
    path = '.'
    main()

