# -*- coding: utf-8 -*-
"""
Module name: SG2T_main
Author:      Ryan Wu
Date:        2020/06/22
Description: Style GAN2 for Waifu generation 
Training time: 11.3 seconds / epoch (@ RTX2080 ti GPU )
Training set:  TWDNE images
Versions:
    V0.10- 2020/06/01: Initial release
    V0.40- 2020/06/12: Modify snapshot update rate from epoch to k*images
                       Modify prediction mechanism: Swap variable between G and G_EMA                       
                       Increase learning rate: 2e-4 -> 8e-4 

Refer to:    
    DCGAN:      "Unsupervised representation learning with deep convolutional 
                  Generative adversarial networks"  by Alec Radford, 2016
    R1_Reg:     "Which Training Methods for GANs do actually Converge?"
                  by Lars Mescheder, 2016
    PG_GAN-     "Progressive growing of GANs for improved quality, stability, and variation"
                  by Tekko Karras
    Style_GAN:  "A style based generator architecture for generative adversarial networks"
                  by Tekko Karras
    Style_GAN2: "Anakyzing and improving the image quality of style GAN"
                  by Tekko Karras
 
                
"""

#---------------------------------------------------- Import libraries
from __future__ import absolute_import, division, unicode_literals, print_function
import tensorflow as tf
import os, time 
import numpy as np
import matplotlib.pyplot as plt
import SG2T_model, SG2T_utils
#import DC_utils, DC_loss, RES_models
 
#---------------------------------------------------- GAN class
class GAN():
    def __init__(self):
        self.epoc        = 0                        # Current epochs
    #---------------------------------------- 
    def train(self,
              epochs     = 35000,                   # Total train image numbers (*1000)
              buffer_size= 64,                      # Training buffer size
              batch_size = 8,                       # Training batch size
              lr         = 2e-4,                    # Learning rate
              ncritic    = 2,                       # = LR(D)  / LR(GS)
              nmapping   = 0.01,                    # = LR(GM) / LR(GS)
              Adam_beta1 = 0.0,                     # Beta_1 parameter for Adam optimizer
              Adam_beta2 = 0.99,                    # Beta_2 parameter for Adam optimizer
              latent_dim = 128,                     # Dimensions of latent space
              dlatent_dim= 128,                     # Dimensions of G mapper output 
              EMA_rate   = 0.9999,                   # Exponential moving average moment
              GP_weight  = 5,                       # Weight of gradient penality
              trun_phi   = 0.7,                     # Truncate phi ratio (1=disable)
              dlatent_a  = 0.01,                    # D latent center update rate
              restart    = False,                   # Training start from scratch?  
              model_name = 'SG2T',                   # File name for model and snapshot
              #path       = '../../../Repository/Anime/TWDNE/TWDNE1/*.jpg',
              path       = '../../../Repository/Anime/TWDNE/*.jpg',
              **_kwargs):
        self.epochs      = epochs                   # Training epochs
        self.buffer_size = buffer_size              # Training set buffer size
        self.batch_size  = batch_size               # Training set batch size
        self.path        = path                     # Training set data path
        self.model_name  = model_name               # Model name 
        self.latent_dim  = latent_dim               # Latent_dim for generator
        self.GP_weight   = GP_weight                # Gradient penalty weight
        self.trun_phi    = trun_phi                 # D latent truncate trick
        self.dlatent_a   = tf.cast(dlatent_a,'float32')      # D latent average alpha
        #--- ---
        self.seed        = tf.random.normal([24, self.latent_dim])   # Seeds for Demo        
        self.G_history   = np.zeros((epochs,))      # History for G loss curve
        self.D_history   = np.zeros((epochs,))      # History for D loss curve
        #self.D_EMA       = SG2T_utils.EMA(EMA_rate)   # EMA        
        self.GS_EMA      = SG2T_utils.EMA(EMA_rate)   # EMA
        self.GM_EMA      = SG2T_utils.EMA(EMA_rate)   # EMA
        self.GS_optimizer= tf.keras.optimizers.Adam(lr,         beta_1= Adam_beta1, beta_2=Adam_beta2)
        self.GM_optimizer= tf.keras.optimizers.Adam(lr*nmapping,beta_1= Adam_beta1, beta_2=Adam_beta2)
        self.D_optimizer = tf.keras.optimizers.Adam(lr*ncritic, beta_1= Adam_beta1, beta_2=Adam_beta2)        
        #--- Declared untrained network ---
        self.GS          = SG2T_model.G_Synthesis(dlatent_size= dlatent_dim)
        self.GM          = SG2T_model.G_mapping(  latent_size= latent_dim, dlatent_size= dlatent_dim)
        self.D           = SG2T_model.D_Style_GAN2()
        dlatent= self.GM(self.seed, training= False) 
        self.dlatent_avg = tf.reduce_mean(dlatent, axis=0)        
        #--- Load previous model---
        if restart == False: self.load_weights() 
        self._train()
    #----------------------------------------  
    def _train(self):  
        print('Begin training')
        ds = SG2T_utils.load_dataset_TWDNE(self.path, self.buffer_size, self.batch_size)    
        a  = 0.99;                                  # History mving average constant
        print('Build dataset')
        start= time.time()
        while self.epoc < (self.epochs-5):
            for r_img in ds: 
              siz     = r_img.shape[0]                # Training batch size              
              if siz == self.batch_size:
                noise   = tf.random.normal([siz, self.latent_dim])
                G_loss, D_loss, dlatent  = self.train_step(r_img, noise) 
                self.GS_EMA.update(self.GS)             # EMA (Exponential moving average)
                self.GM_EMA.update(self.GM)             # EMA (Exponential moving average)
                #self.D_EMA.update(self.D)               # EMA
                #--- dlatent average ---
                new_dlatent_avg = tf.reduce_mean(dlatent, axis=0)
                del_dlatent_avg = new_dlatent_avg - self.dlatent_avg
                self.dlatent_avg= self.dlatent_avg + del_dlatent_avg* self.dlatent_a
                #--- show result ---
                if np.round(self.epoc + siz/1000) > np.round(self.epoc):
                    duration = time.time()-start
                    start    = time.time()
                    F= int(np.round(self.epoc))
                    if F == 0: 
                        self.D_history[0] = D_loss*0.9
                        self.G_history[0] = G_loss*0.9
                    else:
                        self.D_history[F] = self.D_history[F-1]* a +D_loss* (1-a)
                        self.G_history[F] = self.G_history[F-1]* a +D_loss* (1-a)
                     
                    print ('Epoch {} = {} sec; G_loss = {}; D_loss = {}'.format(F +1, 
                                                                        duration, 
                                                                        self.G_history[F], 
                                                                        self.D_history[F]))
                    #--- Show history and save weights ---
                    fn= self.model_name+'X_'+ str(F+1)
                    if (F+1)% 200 == 0: self.save_weights(fn)
                    if (F+1)% 5   == 0: self.save_weights()
                    if (F+1)% 100 == 0: self.show_history()            
                    #--- Show fake image genearted by generator ---
                    if (F+1)% 20  == 0 or F<30: 
                        self.predict(mode ='normal', epoch= F+1)
                    if (F+1)% 200 == 0: 
                        self.predict(mode ='normal', epoch= F+1, save= True)                    
                #--- ---
                self.epoc += siz/1000                       # How many images are trained        
         
    
    #---------------------------------------- train one step (NS loss + R1 regular)
    @tf.function
    def train_step(self, r_img, noise):
        BCE = tf.keras.losses.BinaryCrossentropy(from_logits= True)
        with tf.GradientTape() as GM_tape, tf.GradientTape() as GS_tape, tf.GradientTape() as D_tape:
            dlatent= self.GM(noise, training= True)
            f_img  = self.GS(dlatent , training= True)          # Fake image by generator            
            r_dout = self.D(r_img, training = True)             # Real image critic output
            f_dout = self.D(f_img, training = True)             # Fake image critic output
            #--- apply R1 regulation on D ---    
            with tf.GradientTape() as X_tape:
                X_tape.watch(r_img)                             # 
                x_dout = self.D(r_img,training= True)           # 
            X_grad = X_tape.gradient(x_dout, r_img)             # Gradient Dx
            ddx    = tf.sqrt(tf.reduce_sum(X_grad ** 2, axis=[1,2,3]))
            X_loss = tf.reduce_mean((ddx - 0.0) ** 2)           # Gradient penality loss 
            #... ...
            D_lossR= BCE(tf.ones_like(r_dout),  r_dout)
            D_lossF= BCE(tf.zeros_like(f_dout), f_dout)    
            D_lossX= self.GP_weight * X_loss
            D_loss = D_lossR + D_lossF
            D_lossT= D_loss  + D_lossX
            #--- Apply path regulation on G ---
            G_loss = BCE(tf.ones_like(f_dout),  f_dout)
    
        #--- Update weighting---            
        GM_grad = GM_tape.gradient(G_loss , self.GM.trainable_variables)
        GS_grad = GS_tape.gradient(G_loss , self.GS.trainable_variables)
        D_grad = D_tape.gradient(D_lossT, self.D.trainable_variables)
        self.GM_optimizer.apply_gradients(zip(GM_grad, self.GM.trainable_variables))
        self.GS_optimizer.apply_gradients(zip(GS_grad, self.GS.trainable_variables))
        self.D_optimizer.apply_gradients( zip(D_grad, self.D.trainable_variables))
        return G_loss, D_loss, dlatent 
    
    #---------------------------------------- save weights
    def save_weights(self, fn = None):
        if fn is None: fn = self.model_name
        a0 = self.GS_EMA.get_weights();             # Shadow variables for GS
        a1 = self.GM_EMA.get_weights();             # Shadow variables for GM
        a2 = self.dlatent_avg;                      # Averaged d latent center (mean face)
        A  = [a0, a1, a2]
        np.save(fn + '_MDL_A.npy', A)
        b0 = self.epoc;                             # Training how many epochs
        b1 = self.seed;                             # Restore seed for demo
        b2 = self.G_history;                        # Generator loss curve
        b3 = self.D_history;                        # Discriminator loss curve
        #b4 = self.D_EMA.get_weights();              # Shadow variables for D
        b4 = self.D.get_weights();              # Shadow variables for D
        B  = [b0, b1, b2, b3, b4]
        np.save(fn + '_MDL_B.npy', B)
   
    #---------------------------------------- load weights        
    def load_weights(self):
        fn = self.model_name
        if os.path.isfile(fn + '_MDL_A.npy'):        
            A = np.load(fn + '_MDL_A.npy', allow_pickle = True)
            self.GS_EMA.set_weights(A[0]); self.GS.set_weights(A[0])
            self.GM_EMA.set_weights(A[1]); self.GM.set_weights(A[1])
            self.dlatent_avg = A[2]
            B = np.load(fn + '_MDL_B.npy', allow_pickle = True)
            self.epoc      = B[0]
            self.seed      = B[1]
            self.G_history = B[2]
            self.D_history = B[3]        
            #self.D_EMA.set_weights(B[4]); 
            self.D.set_weights(B[4])
        else:
            print('Model file not exist')

    #---------------------------------------- show loss history 
    def show_history(self, D= True, G= True):
        F = int(np.floor(self.epoc))
        x = np.arange(0, F)
        if D == True: plt.plot(x,self.D_history[0:F],label='D_loss')
        if G == True: plt.plot(x,self.G_history[0:F],label='G_loss')
        plt.legend(); plt.show()        
        
    #---------------------------------------- predict
    def predict(self, taps= 24, mode='random',  epoch= -1, save= False):
        """ mode = 'random': Image from random seed
            mode = 'normal': Image from built-in seed
            mode = 'intrep': Image with interpolation 
        """
        seed = self.seed
        if mode == 'random': seed = tf.random.normal([24, self.latent_dim])
        if mode == 'interp': 
            seed = tf.random.normal([24, self.latent_dim]) 
            seed = seed.numpy()
            for kx in range(1,5):
                for ky in range(4):
                    A= seed[ky*6,:]; B=seed[ky*6+5,:];
                    seed[ky*6+kx,:] = (A*(5-kx) + B*kx)/5 
            seed= tf.cast(seed, tf.float32)
        #--- Load model file if model not exist ---
        if self.epoc == 0.0: self.load_weights()    
        if self.epoc == 0.0:
            print('Please train GAN network first')
        else:    
            prediction = self._EMA_predictor(seed)
            if mode=='randomx':
                dout = self.D(prediction, training = False) 
                print(np.log(-dout.numpy()[1]))
                doutx= np.reshape(dout.numpy(),(4,6))
                print(np.log(-doutx))
                
            self._draw_fake_image(prediction, taps = taps, epoch = epoch, save=save)
 
    
    #---------------------------------------- private function
    def _EMA_predictor(self, seed):                     #-- Predict by EMA shadow generator
        WGS_EMA = self.GS_EMA.get_weights();            # EMA weights
        WGM_EMA = self.GM_EMA.get_weights();            # EMA weights
        if WGS_EMA is not None:                         # Extract EMA weightings
            WGS = self.GS.get_weights()
            WGM = self.GM.get_weights()
            self.GS.set_weights(WGS_EMA)
            self.GM.set_weights(WGM_EMA)
        dlatent    = self.GM(seed,    training=False)   
        #--- D latent space truncate trick ---
        mean_face   = self.dlatent_avg 
        del_dlatent = dlatent - mean_face
        dlatent_new = mean_face + self.trun_phi* del_dlatent        
        #--- ---
        prediction = self.GS(dlatent_new, training=False)   
        if WGS_EMA is not None:                         # Restore GM, GS weightings
            self.GS.set_weights(WGS)
            self.GM.set_weights(WGM)
        return prediction
        
    #........................................ Show generated images
    def _draw_fake_image(self, prediction, taps = 24, epoch = -1, save = False):  
        fig  = plt.figure(figsize=(18,12))
        rows = 4 ; cols = 6;
        if taps == 12: rows = 3; cols = 4
        if taps ==  6: rows = 2; cols = 3
        if taps ==  2: rows = 1; cols = 2
        if taps ==  1: rows = 1; cols = 1
        for i in range(taps):
            plt.subplot(rows, cols, i+1)
            imgs= np.array(prediction[i,:,:,:]*0.5 + 0.5) 
            imgs= np.minimum(np.maximum(imgs,0),1)
            plt.imshow(imgs);
            if i==1 and epoch>-1:
                plt.title('{}: Train {} x 1000 images'.format(self.model_name,epoch))
                plt.xticks([]); plt.yticks([])
            else:
                plt.axis('off')
        #--- save figure ---
        if save is True:
            file_name = self.model_name + '_{:04d}.png'.format(epoch)
            plt.savefig(file_name)
        plt.show()
    #........................................    
  
#---------------------------------------------------- 
#---------------------------------------------------- 
model = GAN()
#model.train(restart= True) 
model.train(restart= False) 
#----------------------------------------------------

#----------------------------------------------------





