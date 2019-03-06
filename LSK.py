#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:29:26 2019

@author: ouhajime
"""

import spacy
import nltk
import pandas as pd
import numpy as np

def cutWords(content):
    #先分句再分词
    sents = nltk.sent_tokenize(content)
    word = []
    for sent in sents:
        word.extend(nltk.word_tokenize(sent))

    return word

nlp = spacy.load('en_core_web_sm')
  
# sent = "The service is absolutely terrible"
# target = 'service'

def getLSK(sent,target):
    words = cutWords(sent)

    sent_len = len(words)

    distance = np.zeros((sent_len,sent_len))

    sent = unicode(sent, "utf-8")


    doc = nlp(sent)
    try:
        for token in doc:
        #     print(token.text, token.head.text,
        #           [child for child in token.children])
            index1 = words.index(token.text)
            index2 = words.index(token.head.text)
            if token.text != token.head.text:
                distance[index1,index2] = 1
            for child in token.children:
                index3 = words.index(str(child))
                distance[index1,index3] = 1
        # print distance

        lsk = np.zeros(sent_len)

        all_target = cutWords(target)
        for t in all_target:
            row = words.index(t)
            lsk = distance[row,:]
            for i,value in enumerate(lsk):
                if value == 1:
                    lsk = lsk + distance[i,:]

        for i,value in enumerate(lsk):
            if value > 1:
                lsk[i] = 1

        return lsk
    except ValueError:
        return np.random.uniform(-0.01,0.01,(sent_len))
    
def saveLSK(mode):
    data_path = './DATA/{}.csv'.format(mode)
    
    df = pd.read_csv(data_path)
    LSK = df.apply(lambda row: getLSK(row['text'], row['target']),axis=1)
    
    save_path = './DATA/{}/LSK.npy'.format(mode)
    np.save(save_path,LSK)
    
if __name__ == '__main__':
    saveLSK('train')
    saveLSK('test')
    
    
    
    
    
    
    
    
   
    
    
    
    
    