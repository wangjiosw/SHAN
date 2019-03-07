#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 18:03:15 2019

@author: ouhajime
"""

import pandas as pd
import nltk
from enchant.checker import SpellChecker
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import keras
import numpy as np
import os



def cutWords(content):
    sents = nltk.sent_tokenize(content)
    word = []
    for sent in sents:
        word.extend(nltk.word_tokenize(sent))

    return word

    

chkr = SpellChecker("en_US")
# stop_words = stopwords.words('english')
# for w in ['!',',','.','?','-s','-ly','</s>','s','(',')','\'','\"']:
#     stop_words.append(w)

stop_words = []
stop_words.extend(['!',',','.','?','-s','-ly','</s>','s','(',')','\'','\"','-'])



wnl = WordNetLemmatizer()


def pre(sent):
    
    # To low case
    sent = sent.lower()
    
    # Spell check
    chkr.set_text(sent)
    
    for err in chkr:
        try:
            sent = sent.replace(err.word,chkr.suggest(err.word)[0])
        except IndexError:
            continue
    
    word_list = cutWords(sent)
    # filter stop words        
    # filtered_words = [word for word in word_list if word not in stop_words]
    
    # Lemmatization
    # lwords = []
    # for w in word_list:
    #     lwords.append(wnl.lemmatize(w))  
    
    
    return word_list


def getLabel(result):
    if result == 'positive':
        return 1
    elif result == 'neutral':
        return 0
    elif result == 'negative':
        return -1
    else:
        print result
        print 'error type'
        exit(1)
  
def savePreData(input_path,save_file_name):
    
    dir_path = './DATA/{}'.format(save_file_name)
    if not os.path.exists(dir_path): 
        os.makedirs(dir_path) 
        
    x_path = './DATA/{}/{}_x.npy'.format(save_file_name,save_file_name)
    target_path = './DATA/{}/{}_target.npy'.format(save_file_name,save_file_name)
    label_path = './DATA/{}/{}_label.npy'.format(save_file_name,save_file_name)

    train_df = pd.read_csv(input_path,encoding='utf-8')
    train_text = train_df.text
    train_target = train_df.target
    train_label = train_df.label
    
    train_label = train_label.apply(getLabel)
    train_label = keras.utils.to_categorical(train_label, num_classes=3)
    train_x = train_text.apply(pre)
    train_target = train_target.apply(pre)
    
    np.save(x_path,train_x)
    np.save(target_path,train_target)
    np.save(label_path,train_label)


if __name__ == "__main__":
    
    savePreData('./DATA/train.csv','train')
    savePreData('./DATA/test.csv','test')













