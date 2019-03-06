#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:00:41 2019

@author: ouhajime
"""

from keras import backend as K
from keras.layers import Layer
from keras import regularizers
import Setting

class GATING_LAYER(Layer):

    def __init__(self, **kwargs):
        super(GATING_LAYER, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.W_gate = self.add_weight(name='W_gate', 
                                      shape=(input_shape[1], input_shape[1]/2),
                                      initializer='glorot_uniform',
                                      regularizer=regularizers.l2(Setting.reg_w),
                                      trainable=True)
        
        super(GATING_LAYER, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
      return K.tanh(K.dot(x,self.W_gate))
        

    def compute_output_shape(self, input_shape):
      
      return (input_shape[0],input_shape[1]/2)
