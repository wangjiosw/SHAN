#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 14:56:17 2019

@author: ouhajime
"""
from keras import backend as K
from keras.layers import Layer
from keras import regularizers
import Setting

class GLOBAL_ATTENTION(Layer):

    def __init__(self, **kwargs):
        super(GLOBAL_ATTENTION, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.W_att1 = self.add_weight(name='W_att1', 
                                      shape=(input_shape[2], input_shape[2]),
                                      initializer='glorot_uniform',
                                      regularizer=regularizers.l2(Setting.reg_w),
                                      trainable=True)
        
        self.W_att2 = self.add_weight(name='W_att2', 
                                      shape=(input_shape[2], 1),
                                      initializer='glorot_uniform',
                                      regularizer=regularizers.l2(Setting.reg_w),
                                      trainable=True)
        
        self.b_att1 = self.add_weight(name='b_att1', 
                                      shape=(1, input_shape[2]),
                                      initializer='glorot_uniform',
                                      regularizer=regularizers.l2(Setting.reg_w),
                                      trainable=True)
        super(GLOBAL_ATTENTION, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        n = K.dot((K.tanh(K.dot(x, self.W_att1)+ self.b_att1)), self.W_att2)
        e_m = K.exp(n)
        count = K.sum(e_m)
        alpha = e_m/count
        return alpha
        

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],1)