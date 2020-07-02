# -*- coding: utf-8 -*-
"""
Module name: SG2_utils
Author:      Ryan Wu
Date:        V0.1- 2020/06/01: Add 
Description: GAN utilies for FFHQ image generation (resolution = 512 x 512)
"""

#---------------------------------------------------- Import libraries
from __future__ import absolute_import, division, unicode_literals, print_function
import tensorflow as tf 

#---------------------------------------------------- load images (FFHQ)
def load_image(path):
    fil   = tf.io.read_file(path);                  # Load file
    img   = tf.image.decode_jpeg(fil);              # Decode as jpeg
    img   = tf.cast(img[:, :, :], tf.float32)
    img   = img/127.5 - 1;
    return img
    #----------------------------------------
def load_image_TWDNE(path, pad =24):
    fil   = tf.io.read_file(path);                  # Load file
    img   = tf.image.decode_jpeg(fil);              # Decode as jpeg 
    img   = tf.image.random_crop(img, [960,960,3])  # Random crop as square
    img   = tf.image.resize(img,(512,512))          # Resize as 512 x 512
    img   = tf.cast(img[:, :, :], tf.float32)       # Convert to tensor
    img   = img/127.5 - 1;                          # Normalize to [-1 ~ +1]    
    return img
    #---------------------------------------- 
def load_dataset_TWDNE(path, buffer_size, batch_size):
    ds    = tf.data.Dataset.list_files(path) 
    ds    = ds.map(load_image_TWDNE)
    ds    = ds.shuffle(buffer_size)
    ds    = ds.batch(batch_size)    
    return ds
    #----------------------------------------   
#---------------------------------------------------- EMA: Exponential average moving
class EMA(object):
    def __init__(self,rate):
        self.rate = rate;                           # 
        self.W    = None;                           # Shadow weights
    #....................................
    def get_weights(self): return self.W            #-- get shadow weights
    def set_weights(self,W): self.W = W;            #-- set shadow weights
    #....................................
    def copy(self, model):                          #-- direct copy all weights from W
        self.W = [];
        W  = model.get_weights(); 
        WL = len(W);                                #
        for k in range(WL): self.W.append(W[k]);
    #---------------------------------------- 
    def update(self, model):                        #-- update weights inside W        
        if self.W is None:
            self.copy(model)
        else:
            W  = model.get_weights(); 
            WL = len(W);
            a  = self.rate;
            for k in range(WL):
                self.W[k] = self.W[k]*a + W[k]*(1-a)    # exponential moving average   
#---------------------------------------------------- Loss function
    
#---------------------------------------------------- 
 
