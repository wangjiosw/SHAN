#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:01:15 2019

@author: ouhajime
"""

from keras import backend as K
from keras.layers import Layer
from keras import regularizers
import Setting

class SOFTMAX_LAYER(Layer):

    def __init__(self, output_dim,**kwargs):
        self.output_dim = output_dim
        super(SOFTMAX_LAYER, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.W_s = self.add_weight(name='W_s', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='glorot_uniform',
                                      regularizer=regularizers.l2(Setting.reg_w),
                                      trainable=True)
        
        self.b_s = self.add_weight(name='b_s', 
                                      shape=(self.output_dim,),
                                      initializer='glorot_uniform',
                                      regularizer=regularizers.l2(Setting.reg_w),
                                      trainable=True)
        
        super(SOFTMAX_LAYER, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
      return K.softmax(K.dot(x,self.W_s)+self.b_s)
        

    def compute_output_shape(self, input_shape):
      
      return (input_shape[0],self.output_dim)