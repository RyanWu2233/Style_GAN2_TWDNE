# -*- coding: utf-8 -*-
"""
Module name: SG2_model
Author:      Ryan Wu
Date:        
    V0.10- 2020/06/01: Initial release
    V0.40- 2020/06/11: Modify as TensorFlow 2.1 Keras API
    V0.42- 2020/06/12: Act @ G_synthesis block: linear  -> ReakyReLU(0.2)
                       Normalize @ ToRGB:       apply   -> remove   
                       Upsampling method:       nearest -> bilinear
 
             
Description: Style GAN2 model for FFHQ generation
Keypoints for Discriminator:
    (D1)  Resnet discriminator to achieve best FID    (section 4.1)
    (D2)  Do not use spectral normalization             (appendix E)
    
    (D)   Minibatch standard deviation layer at end     (appendix B)
    -----------------------------------------------------------------
    (G1)  Skip net generator to achieve best PPL        (section 4.1)
    (G2)  Weight demodulation to remove droplet         (section 2.2)
    (G3)  Non-saturation logistic loss                  (appendix B)
    (G4)  R1 regularization                             (B:)
    (G5)  Grouped convolutions                          (B:)
    (G6)  Do not use spectral normalization             (appendix E)
    
    (G)   Constant input for generator                  (style GAN)
    (G)   Latent dimensionality = 512                   (style GAN)
    (G)   8 layers mapping netwrok Z -> W               (style GAN)
    (G)   Equalized learning rate for all trainable W   (style GAN)
    (G)   Leaky ReLU with alpha = 0.2                   (style GAN)
    (G)   Bilinear filtering for all up/down layers     (style GAN)
    (G)   Exponential moving average                    (style GAN)
    (G)   Style mixing regularization                   (style GAN)
    
"""



#---------------------------------------------------- Import libraries
from __future__ import absolute_import, division, unicode_literals, print_function
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, LeakyReLU, Layer, UpSampling2D
from tensorflow.keras.layers import Input, AveragePooling2D, Add, Dense
import numpy as np

#----------------------------------------------------  Basic layer
def Conv2DA(fmaps, kernel= 3, stride= 1, padding='same', use_bias= True):
    m   = tf.keras.Sequential()
    m.add(Conv2D(fmaps, kernel, stride, padding=padding, use_bias= use_bias))
    m.add(LeakyReLU(0.2))
    return m
    #................................................    
def DenseA(fmaps, use_bias = True):
    m   = tf.keras.Sequential()
    m.add(Dense(fmaps, use_bias= use_bias))
    m.add(LeakyReLU(0.2))
    return m
    #................................................    
class D_Minibatch_Stdev(Layer):
    def __init__(self, group_size):
        super(D_Minibatch_Stdev, self).__init__()
        self.group_size = group_size
    def build(self, input_shape):
        pass
    def call(self, inputs):
        s = tf.shape(inputs)
        group_size = tf.minimum(self.group_size, s[0])      # Group size should no larger than batch size
        y = tf.reshape(inputs, [group_size, -1, s[1],s[2],s[3]]) # [GMHWC]
        y = tf.cast(y,tf.float32)
        y = y - tf.reduce_mean(y, axis=0, keepdims = True)      # [.MHWC] Subtract mean over group
        y = tf.reduce_mean(tf.square(y),axis=0)                 # [ MHWC] Calc variance over group
        y = tf.sqrt(y+ 1e-8)                                    # [ MHWC] Calc stddev over group
        y = tf.reduce_mean(y, axis=[1,2,3], keepdims=True)      # [ M...] Take average over fmaps and pixels
        y = tf.cast(y, inputs.dtype)
        y = tf.tile(y, [group_size,s[1],s[2],1])                # [ NHW1] Replica over group and pixels
        z = tf.concat([inputs,y],axis=3)                        # [NHW1] & [NHWC]
        return z
    #................................................    
def D_Style_GAN2(
        fmap_max        = 256//2,                   # Maximum feature map number
        fmap_min        = 16,                       # Minimum feature map number
        resolution      = 512,                      # Resolution of training images
        **_kwargs):
    res_log2 = int(np.log2(resolution))             # = log2(resolution)
    seq = np.arange(res_log2-1,-1,-1)               # = [9,8,7,6,5,4,3,2,1,0]
    nf  = np.minimum(fmap_min*(2**seq), fmap_max)   # = [256,256,256,256, .. 128,64,32]
    #--- Resnet stage of D---
    def D_Resnet(fin, fout):
        x0  = Input(shape=(None,None,fin))             # Input = [N,H,W,Cin]
        x1  = Conv2DA(fin)(x0)
        x2  = Conv2DA(fout)(x1)
        x3  = AveragePooling2D()(x2)    
        y1  = Conv2D(fout, 1, 1, padding='same', use_bias= False)(x0)    
        y2  = AveragePooling2D()(y1)
        x3  = Add()([x3,y2])
        return Model(x0,x3)  
    #--- Main net ---
    x0  = Input(shape=(None,None,3))                # out= [512,512, 3]
    x1  = Conv2DA(nf[8], 1, 1)(x0)                  # out= [512,512, nf[8]]    
    x1  = D_Resnet(nf[8],nf[7])(x1)                 # out= [256,256, nf[7]]
    x1  = D_Resnet(nf[7],nf[6])(x1)                 # out= [128,128, nf[6]]
    x1  = D_Resnet(nf[6],nf[5])(x1)                 # out= [ 54, 64, nf[5]]
    x1  = D_Resnet(nf[5],nf[4])(x1)                 # out= [ 32, 32, nf[4]]
    x1  = D_Resnet(nf[4],nf[3])(x1)                 # out= [ 16, 16, nf[3]]
    x1  = D_Resnet(nf[3],nf[2])(x1)                 # out= [  8,  8, nf[2]]
    x2  = D_Resnet(nf[2],nf[1])(x1)                 # out= [  4,  4, nf[1]]
     
    x2  = D_Minibatch_Stdev(4)(x2)                  # out= [  4,  4, nf[1]+1]  
    x2  = Conv2DA(nf[1], 3, 1)(x2)                  # out= [  4,  4, nf[1]]
    x2  = Conv2DA(nf[0], 4, 1, padding='valid')(x2) # out= [  1,  1, nf[0]]
    x3  = Dense( 1, use_bias=True)(x2)              # out= [  1,  1, 1]
    return Model(x0, x3)

#----------------------------------------------------  Style GAN2 Generator (space transform)
def G_mapping(
        latent_size     = 512//4,                   # Latent vector (Z) dimensionality.
        mapping_fmaps   = 512//4,                   # Number of activations in the mapping layers.
        dlatent_size    = 512//4,                   # Disentangled latent (W) dimensionality.
        **_kwargs):    
    x0  = Input(shape=(latent_size,))
    n0  = tf.reduce_mean(tf.square(x0),axis=1, keepdims = True) # Normalize
    x1  = x0 * tf.math.rsqrt(n0 + 1e-8)   
    x1  = DenseA(mapping_fmaps)(x1)
    x1  = DenseA(mapping_fmaps)(x1)                 # Layer 1
    x1  = DenseA(mapping_fmaps)(x1)                 # Layer 2
    x1  = DenseA(mapping_fmaps)(x1)                 # Layer 3
    x1  = DenseA(mapping_fmaps)(x1)                 # Layer 4
    x1  = DenseA(mapping_fmaps)(x1)                 # Layer 5
    x1  = DenseA(mapping_fmaps)(x1)                 # Layer 6
    x2  = DenseA(dlatent_size )(x1)                 # Layer 7
    return Model(x0,x2)    
        
#---------------------------------------------------- Style GAN2 modulator
# AdaIN stage in Style GAN2 paper figure 2.(c)
# Input sources are: (0) image data and (1) Disentangle latent space
class Conv2DM(Layer):    
    def __init__(self, fmaps, kernel = 3, stride = 1, 
                 apply_noise= True,                 #
                 demodulate = True):                #
        super(Conv2DM, self).__init__()
        self.fmaps = fmaps
        self.k     = kernel
        self.stride= stride
        self.apply_noise = apply_noise
        self.demodulate  = demodulate
    def build(self, input_shape):
        fin     = input_shape[0][3]                 # Number of input  feature channels
        fout    = self.fmaps                        # Number of output feature channels
        fdlatent= input_shape[1][1]                 # Number of D latent feature channels
        #-- Conv 2D weight --        
        self.w_conv = self.add_weight(shape = (self.k, self.k, fin, fout),
                                      initializer='random_normal',
                                      trainable=True,
                                      dtype = tf.float32)
        self.b_conv = self.add_weight(shape = (1,1,fout),
                                      initializer='zeros',
                                      trainable=True)        
        #-- Block A () weight and bias --
        self.w_a    = self.add_weight(shape = (fdlatent, fin),
                                      initializer='random_normal',
                                      trainable=True)
        self.b_a    = self.add_weight(shape = (fin,),
                                      initializer='zeros',
                                      trainable=True)  
        #-- Block B () weight --
        if self.apply_noise is True:
            self.n_str  = self.add_weight(shape=(1,),
                                      initializer='random_normal',
                                      trainable=True)         
    def call(self, inputs):
        #-- Block A --
        din  = inputs[1]                            # Input from disentangle latent
        s    = tf.matmul(din, self.w_a)+ self.b_a   # s = style gain, format= [BI]
        ss   = tf.expand_dims( s, axis = 1)         # [BI]   -> [B.I]
        ss   = tf.expand_dims(ss, axis = 2)         # [B.I]  -> [B..I]
        
        #-- Modulate and Conv --
        cin  = inputs[0]                            # Input from pevious stage
        xm   = cin * tf.cast(ss, cin.dtype)         # Modulate style factor
        xs   = tf.nn.conv2d(xm, tf.cast(self.w_conv, cin.dtype), data_format='NHWC', strides=[1,1,1,1], padding='SAME')
        #-- Normalize --
        if self.demodulate == True:
            d  = tf.reduce_sum(tf.square(self.w_conv), axis= [0,1,2]) # [O]
            dd = tf.math.rsqrt(d + 1e-8)            # = 1/sqrt(sum(ww^2)) = 1/sigma
            dd = tf.expand_dims(dd, axis = 0)       # -> [.O]
            dd = tf.expand_dims(dd, axis = 1)       # -> [..O]
            dd = tf.expand_dims(dd, axis = 2)       # -> [...O]
            xs = xs  * tf.cast(dd, cin.dtype)       # Normalize scale
        #-- Block B --
        xb   = xs + self.b_conv                     # Add bias
        if self.apply_noise is True:                # Add noise
            s_cin= tf.shape(cin)
            noise= tf.random.normal( shape= (s_cin[0], s_cin[1], s_cin[2], 1))
            xb = xb + self.n_str* noise
        return   xb
#----------------------------------------------------  Head of generator synethsis network 
class G_Header(Layer):    
    def __init__(self, fmaps):
        super(G_Header, self).__init__()
        self.fmaps = fmaps
    def build(self, input_shape):         
        self.w_conv = self.add_weight(shape = (1, 4, 4, self.fmaps),
                                      initializer='random_normal',
                                      trainable=True,
                                      dtype = tf.float32)
    def call(self, inputs):
        s_in = tf.shape(inputs)
        x = tf.tile(self.w_conv, [s_in[0], 1, 1, 1])     # Duplicate fmaps time
        return x
#----------------------------------------------------  Style GAN2 Synthesis network 
def G_Synthesis(
        fmap_max        = 256//2,                   # Maximum feature map number
        fmap_min        = 8,                       # Minimum feature map number
        resolution      = 512,                      # Resolution of training images
        dlatent_size    = 512//4,                   # Disentangled latent (W) dimensionality.
        #upsample_method = 'nearest',                # Upsampling method
        upsample_method = 'bilinear',                # Upsampling method
        **_kwargs):
    res_log2 = int(np.log2(resolution))             # = log2(resolution)
    seq = np.arange(res_log2-1,-1,-1)               # = [9,8,7,6,5,4,3,2,1,0]
    nf  = np.minimum(fmap_min*(2**seq), fmap_max)   # = [256,256,256,256, .. 128,64,32]    
    #--- Block ---
    def G_Block(fin, fout):
        din = Input(shape = (dlatent_size,))        # D latent space input
        cin = Input(shape = (None, None, fin))      # Image input
        c1  = Conv2DM(fout)([cin, din])
        c1  = LeakyReLU(0.2)(c1)
        c2  = UpSampling2D(interpolation = upsample_method)(c1)
        c2  = Conv2DM(fout)([c2, din])
        c2  = LeakyReLU(0.2)(c2)
        return Model([cin,din], c2)
    #--- ToRGB ---    
    def ToRGB(fin):
        din = Input(shape = (dlatent_size,))        # D latent space input
        cin = Input(shape = (None, None, fin))      # Image input
        c2  = Conv2DM(3, kernel = 1, apply_noise= False, demodulate= False)([cin, din])        
        return Model([cin, din], c2)    
    #--- Main net ---
    din = Input(shape = (dlatent_size,))            # 
    x   = G_Header(nf[1])(din)                      # [  4,  4,  nf[1]]
    x   = Conv2DM(nf[1])([x,din])                   # [  4,  4,  nf[1]]
    x   = LeakyReLU(0.2)(x)
    y   = ToRGB(  nf[1])([x,din])                   # RGB output
    for res in range(3, 10):
        x = G_Block(nf[res-2], nf[res-1])([x,din])
        t = ToRGB(             nf[res-1])([x,din])
        y = UpSampling2D(interpolation = upsample_method)(y)
        y = y + t
    images_out = tf.identity(y)    
    
    return Model(din, images_out)
#---------------------------------------------------- 
 
