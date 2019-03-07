#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:02:20 2019

@author: ouhajime
"""

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.layers import Masking, Reshape, Flatten, Masking 

from keras.layers import Bidirectional
from keras.initializers import glorot_uniform

from keras import backend as K
from keras.layers import Layer
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Lambda, Concatenate, Dot, Multiply, Add, Dropout
from keras import regularizers
import keras

import Setting
from GlobalAttention import GLOBAL_ATTENTION
from SyntaxDirectedLocalAttenyion import SYNTAX_DIRECTED_LOCAL_ATTENTION
from GatingLayer import GATING_LAYER
from SoftmaxLayer import SOFTMAX_LAYER



def getR(ip):
    a = ip[0]
    h = ip[1]
    # wait...
    return K.sum(K.dot(a,h),axis=1) 
  

def getGama(ip):
    a = ip[0]
    b = ip[1]
    g = ip[2]
    h = ip[3]
    
    a = K.repeat(a, 600)
    b = K.repeat(b, 600)
    g = K.repeat(g, 80)
    
    a = tf.transpose(a,[0,2,1])
    b = tf.transpose(b,[0,2,1])

    t = K.relu(a + b*g)
    
    sum_t = K.sum(t,axis=1)
    sum_t = K.repeat(sum_t, 80)

    r = t/sum_t    
    r_final = K.sum(h*r,axis=1)
    
    return r_final
  

def SHAN_Model():
  # INPUT LAYER
  ## text
  input1 = Input(shape=(Setting.max_len,300))
  ## target
  input2 = Input(shape=(1,300))
  ## LSK
  input3 = Input(shape=(Setting.max_len,1))

  
  # MEMORY MODELLING LAYER
  ## Glorot uniform initializer, also called Xavier uniform initializer.
  H = Bidirectional(LSTM(300, return_sequences=True,
                               kernel_regularizer=regularizers.l2(Setting.reg_w),
                               kernel_initializer='glorot_uniform',
                               recurrent_initializer='glorot_uniform',
                               bias_initializer='glorot_uniform'), weights='glorot_uniform')(input1)
  # HYBRID ATTENTION LAYER
  ## GLOBAL ATTENTION
  V_a = Lambda(lambda x: tf.tile(x, multiples=[1,Setting.max_len, 1]))(input2)
  HV = Concatenate()([H,V_a])
  alpha = GLOBAL_ATTENTION()(HV)
  alpha = Reshape((Setting.max_len,))(alpha)
  r_glo = Lambda(getR)([alpha,H])
  
  ## SYNTAX-DIRECTED LOCAL ATTENTION
  exp_n = SYNTAX_DIRECTED_LOCAL_ATTENTION()(HV)
  exp_n_i = Multiply()([exp_n,input3])
  beta =Lambda(lambda x: x/K.sum(x))(exp_n_i)
  
  beta = Reshape((Setting.max_len,))(beta)
  r_loc = Lambda(getR)([beta,H])
  
  # GATING LAYER
  GLO_LOC = Concatenate()([r_glo,r_loc])
  g = GATING_LAYER()(GLO_LOC)
  r = Lambda(getGama)([alpha,beta,g,H])
  
  # SOFTMAX LAYER
  r = Dropout(0.5)(r)
  y_pre = SOFTMAX_LAYER(3)(r)
  
  model = Model(inputs=[input1,input2,input3], outputs=y_pre)  
  
  adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

  model.compile(loss='categorical_crossentropy',
                optimizer=adam,
                metrics=['categorical_accuracy'])
 
  return model

if __name__ == "__main__":
    model = SHAN_Model()
    model.summary()
    
    
    