from underthesea import word_tokenize, sent_tokenize, pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity 
from scipy.spatial import distance 
from collections import defaultdict, OrderedDict 
from string import punctuation
import re
import numpy as np 
import glob

def load_data(path):
  '''@return list of document in specific path'''
  data=[]
  all_files = glob.glob(path + "/*.txt")
  for file in all_files:
      passage=open(file, "r", encoding='utf-8').read()
      data.append(passage)
  return data

def vi_tokenizer(row):
  '''vietnamese tokenizer @return list term'''
  return word_tokenize(row, format="text")

def remove_stopwords(stopwords,text):
  '''remove vietnamese stopwords @param stopwords is set of stopwords, @param text is sentence. @return string'''
  sent = [s for s in text.split() if s not in stopwords ]
  sent = ' '.join(sent)
  return sent


def standardize_data(list_data,stopwords):
  '''standardize data @param list_data is list data, @param stopwords is set of stopwords @return list of standard data'''
  hl_cleansed=[]
  remove = punctuation
  pattern = "[{}]".format(remove) # create the pattern
  re_space=re.compile('\s+')
  re_trailing=re.compile('^\s+|\s+?$')
  for data in list_data:
      data=re.sub(pattern, " ", data) 
      data=re.sub(re_space,' ',data)
      data=re.sub(re_trailing,' ',data)
      data = data.strip()
      data = remove_stopwords(stopwords,data)
      hl_cleansed.append(data)
  return hl_cleansed

def keywords_extraction(sent):
  '''A native function to define good tokens for searching 
        currently get all but except P and CH'''
  rs=""
  for i in pos_tag(sent):
    if i[1] !='P' and i[1] != 'CH':
      rs=rs+' '+i[0]
  return rs.strip()


def arr_convert_1d(arr): 
  '''Converting 3D array of array into 1D array '''
  arr = np.array(arr) 
  arr = np.concatenate( arr, axis=0 ) 
  arr = np.concatenate( arr, axis=0 ) 
  return arr 
  
def cosine(trans):
  '''Cosine Similarity''' 
  cos = [] 
  cos.append(cosine_similarity(trans[0], trans[1])) 
  return cos


def tfidf(str1, str2,tf_idf_vetor): 
  '''Caculate score between @str1 and @str2 using @tf_idf_vector .@return score'''

  corpus = [str1,str2] 
  trans = tf_idf_vetor.transform(corpus)
  cos=cosine(trans) 
  return arr_convert_1d(cos)[0]

def relevance_ranking(query,data,tf_idf_vetor,top_n_matching=None):
  '''relevance ranking between @query and @data using @tf_idf_vector @return top_n_matching if top_n_matching != None, otherwise return all results matching'''
  key_words_query=keywords_extraction(query)
  results=[]
  for doc in data:
    rs=dict()
    ir_score=tfidf(key_words_query, doc,tf_idf_vetor)
    num_overlap=len(set(key_words_query.split()) & set(doc.split()))
    if ir_score!=0.0:
      rs['overlap_key_words']=num_overlap
      rs['ir_score']=ir_score
      rs['content']=doc
      results.append(rs)
  results=sorted(results, key=lambda k: k['ir_score'], reverse=True)
  if top_n_matching:
    results=results[:top_n_matching]
  return results


def IR_QA(query,data,model,tf_idf_vetor,top_n_matching=None):
  '''
    etract answer from data.
  '''
  key_words_query=keywords_extraction(query)
  IR_ranking=relevance_ranking(key_words_query,data,tf_idf_vetor,top_n_matching)
  rs=[]
  for ranking in IR_ranking :
    result=dict()
    answer = model.predict(ranking['content'],query)
    if answer['confidence'] >0.4:
      result['overlap_key_words']=ranking['overlap_key_words']
      result['answer']=answer['answer']
      result['ir_score']=ranking['ir_score']
      result['bert_score']=answer['confidence']
      result['content']=ranking['content']
      rs.append(result)
  return rs 