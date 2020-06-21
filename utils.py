from underthesea import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity 
from scipy.spatial import distance 
from collections import defaultdict, OrderedDict 
from string import punctuation
import unidecode
import re
import pandas as pd 
import numpy as np 
import glob
from bert import QA
import time
import sys
import os
from underthesea import pos_tag


def load_data(path):
  data=[]
  all_files = glob.glob(path + "/*.txt")
  for file in all_files:
      passage=open(file, "r", encoding='utf-8').read()
      data.append(passage)
  return data

def vi_tokenizer(row):
    return word_tokenize(row, format="text")

def remove_stopwords(stopwords,text):
  sent = [s for s in text.split() if s not in stopwords ]
  sent = ' '.join(sent)
  return sent


def remove_punctuation(row):
  remove = punctuation
  remove = remove.replace("_", "")
  pattern = "[{}]".format(remove) # create the pattern
  re_space=re.compile('\s+')
  re_trailing=re.compile('^\s+|\s+?$')
  row=re.sub(pattern, " ", row) 
  row=re.sub(re_space,' ',row)
  row=re.sub(re_trailing,' ',row)
  row = row.strip()
  row =row.lower()
  return row

def standardize_data(df,stopwords):
    hl_cleansed=[]
    remove = punctuation
    #remove = remove.replace("_", "")
    pattern = "[{}]".format(remove) # create the pattern
    re_space=re.compile('\s+')
    re_trailing=re.compile('^\s+|\s+?$')
    for row in df:
        #row = vi_tokenizer(row)
        row=re.sub(pattern, " ", row) 
        row=re.sub(re_space,' ',row)
        row=re.sub(re_trailing,' ',row)
        row = row.strip()
        row = remove_stopwords(stopwords,row)
        #row = remove_accents(row)
        #row = row.lower()
        hl_cleansed.append(row)
    return hl_cleansed

def keywords_extraction(sent):
  rs=""
  for i in pos_tag(sent):
    if i[1] !='P' and i[1] != 'CH':
      rs=rs+' '+i[0]
  return rs.strip()


def sentences_tokenize(text):
    sents = sent_tokenize(text)
    sents = [word_tokenize(s,format = 'text') for s in sents]
    sents = [remove_punctuation(s) for s in sents]
    sents = [s.lower() for s in sents]
    #sents = [remove_stopwords(stopwords,s) for s in sents]
    return sents

## Converting 3D array of array into 1D array 
def arr_convert_1d(arr): 
    arr = np.array(arr) 
    arr = np.concatenate( arr, axis=0 ) 
    arr = np.concatenate( arr, axis=0 ) 
    return arr 
  
## Cosine Similarity 
def cosine(trans): 
    cos = [] 
    cos.append(cosine_similarity(trans[0], trans[1])) 
    return cos

def tfidf(str1, str2,tf_idf_vetor,stopwords):
    str1=standardize_data([str1],stopwords)
    str2=standardize_data([str2],stopwords)  
    corpus = [str1[0],str2[0]] 
    trans = tf_idf_vetor.transform(corpus)
    cos=cosine(trans) 
    return arr_convert_1d(cos)[0]

def relevance_ranking(query,data,vect,stopwords):
  query=standardize_data([query],stopwords)[0]
  print('Query: ',query,'\n')
  score=defaultdict()
  i=0
  for d in data:
    t=tfidf(query, d,vect,stopwords)
    if t!=0.0:
      score[t]=d
    i+=1
  return OrderedDict(sorted(score.items(),reverse=True))
