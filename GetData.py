#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 17:44:50 2019

@author: ouhajime
"""

import xml.dom.minidom
import pandas as pd

def extractData(file_path):
    """
    extract data from xml file
    """
    
    DOMTree = xml.dom.minidom.parse(file_path)
    collection = DOMTree.documentElement
    
    data = []
    sents = collection.getElementsByTagName("sentence") 
    for sent in sents:
      aspectTerms = sent.getElementsByTagName('aspectTerms')
      if len(list(aspectTerms)):
        
        text = sent.getElementsByTagName("text")[0]
        temp = text.childNodes[0].data
    
        aspectTerm = aspectTerms[0].getElementsByTagName("aspectTerm")
        for ap in aspectTerm:
          content = []
          content.append(temp)
          content.append(ap.getAttribute("term"))
          content.append(ap.getAttribute("polarity"))
          data.append(content)
    
    df = pd.DataFrame(data,columns=['text','target','label'])
    df = df[df['label'] != 'conflict']
    
    return df

def saveData(input_path,save_path):
    df = extractData(input_path)
    df.to_csv(save_path,index=False)

if __name__ == "__main__":

    saveData('./DATA/Restaurants_Train.xml','./DATA/train.csv')
    saveData('./DATA/restaurants-trial.xml','./DATA/test.csv')








