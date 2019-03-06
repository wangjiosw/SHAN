#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:09:40 2019

@author: ouhajime
"""

import sys
from SHAN import SHAN_Model
import keras
import os
from keras.preprocessing.sequence import pad_sequences
import Setting
import numpy as np


def main(argv):
    
    mode = 'train'

    if len(sys.argv) == 2:
        mode = argv[1]
    
    x_path = "./DATA/{}/glove_vec.npy".format(mode)
    label_path = "./DATA/{}/{}_label.npy".format(mode,mode)
    LSK_path = "./DATA/{}/LSK.npy".format(mode)
    glove_target_vec_path = "./DATA/{}/glove_target_vec.npy".format(mode)

  
    x = np.load(x_path)
    label = np.load(label_path)
    LSK = np.load(LSK_path)
    target = np.load(glove_target_vec_path)
    
    LSK = pad_sequences(LSK, maxlen=Setting.max_len,  dtype='float', padding='post')
    x = pad_sequences(x, maxlen=Setting.max_len,  dtype='float', padding='post')
    
    target = np.reshape(target,(-1,1,300))
    LSK = np.reshape(LSK,(-1,Setting.max_len,1))
    
    model = SHAN_Model()
    
    model_weights_path = './Model/SHAN.h5'
    if os.path.exists(model_weights_path): 
        model.load_weights (model_weights_path)
    
    if mode == 'test':
        print model.evaluate([x,target,LSK], label,batch_size=32)

    elif mode == 'train':
        checkpointer = keras.callbacks.ModelCheckpoint(filepath=model_weights_path, verbose=1, save_best_only=True,save_weights_only=True)
        earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00000001, patience=3, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
        model.fit([x,target,LSK], label,batch_size=128, epochs=31,validation_split=0.2,callbacks=[checkpointer,earlystop])
        # model.save(model_weights_path)

    else:
        print('Input Mode Error !')
    
    


if __name__ == "__main__":
    main(sys.argv)