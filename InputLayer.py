#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 21:28:21 2019

@author: ouhajime
"""

import numpy as np


corpus = []
glove_model_path = './GLOVE_MODEL/glove300d.txt'

with open(glove_model_path, 'r') as f:
    for line in f:
        c = line.split(" ")[0]
        corpus.append(c)
    f.close()
    
def getline(line_num):
    # row num begin from 1 not 0
    if line_num < 1 :return ''
    for currline,line in enumerate(open(glove_model_path,'rU')):
        if currline == line_num -1 : return line
    return ''

def gloveVec(inp):
    """
    transform each word(sentence) to Glove vector(300 dim)
    """
    glove_vecs = [] 
    for sent in inp:
        glove_vec = []
        for w in sent:
            try:
                index = corpus.index(w)
                s = getline(index+1)
                vec = s.split(" ")[1:]
                vec = list(np.array(vec).astype(np.float))
            except ValueError:
                vec = list(np.random.uniform(-0.01,0.01,(300)))
            glove_vec.append(vec)
                
        glove_vecs.append(glove_vec)
    return glove_vecs

def gloveTargetVec(inp):
    """
    transform each word(target) to Glove vector(300 dim)
    """
    vec = np.zeros(300)

    for w in inp:
        try:
            index = corpus.index(w)
            s = getline(index+1)
            s = np.array(s.split(" ")[1:])
            s = s.astype(np.float)
            vec = vec + s
        except ValueError:
            vec = vec + np.random.uniform(-0.01,0.01,(300))
    
    vec =  vec/len(inp)
    return list(vec)

def saveGloveVec(mode):
    # mode = 'train' or 'test'
    train_x_path = './DATA/{}/{}_x.npy'.format(mode,mode)
    train_target_path = './DATA/{}/{}_target.npy'.format(mode,mode)
    
    train_x = np.load(train_x_path)
    train_target = np.load(train_target_path)

    glove_vec = gloveVec(train_x)
    
    glove_vec_path = "./DATA/{}/glove_vec.npy".format(mode)
    np.save(glove_vec_path,glove_vec)
    
    # golve_target_vec = train_target.apply(gloveTargetVec)
    t_len = len(train_target)
    golve_target_vec = []
    for i in range(t_len):
        vec = gloveTargetVec(train_target[i])
        golve_target_vec.append(vec)
    
    glove_target_vec_path = "./DATA/{}/glove_target_vec.npy".format(mode)
    np.save(glove_target_vec_path,golve_target_vec)

    

if __name__ == "__main__":
    
    saveGloveVec('train')
    saveGloveVec('test')






