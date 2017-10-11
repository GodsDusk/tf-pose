#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imresize 


joint_list = ['r ankle', 'r knee', 'r hip', 'l hip', 'l knee', 'l ankle', 
                'pelvis', 'thorax', 'upper neck', 'head top', 
                'r wrist', 'r elbow', 'r shoulder', 'l shoulder', 'l elbow', 'l wrist']

def view(image, heatmaps, show_max):
    # print heatmaps.shape
    image = imresize(image, (heatmaps.shape[0], heatmaps.shape[1]))
    for i in xrange(heatmaps.shape[2]):
        joint_image = np.zeros_like(image)
        plt.suptitle('output')
        plt.subplot(4,4,i + 1)
        heatmap = heatmaps[:,:,i]
        joint = np.where(heatmap==np.max(heatmap))
        plt.title(joint_list[i])
        # plt.title(str(np.max(heatmap)))
        if show_max and len(joint[1]) < 10:
             plt.plot(joint[1], joint[0], 'r.')    
        plt.imshow(image)
        plt.imshow(heatmap * 255, alpha=0.5)  
    plt.show()


