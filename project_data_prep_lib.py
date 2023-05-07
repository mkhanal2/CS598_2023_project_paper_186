import pandas as pd
import os
import numpy as np
import torch

import spacy
from gensim.models import Word2Vec

WORD_2_VEC_MODEL_PATH = "./drive/MyDrive/DLHC-Data/word2vec.model"

med7 = spacy.load("en_core_med7_lg")
w2vec = Word2Vec.load(WORD_2_VEC_MODEL_PATH)

def data_prep_time_series(data):
    idx = pd.IndexSlice
    out = data.copy()
    GRP_BY_COL = ['subject_id', 'hadm_id', 'icustay_id'] #This colum is used for group-by
    #out = out.loc[:, idx[:, ['mean', 'count']]] #Only take mean and count column.  
    out = out.loc[:, idx[:, ['mean']]] #Only take mean
    icustay_means = out.loc[:, idx[:, 'mean']].groupby(GRP_BY_COL).mean() #calcuate mean for different hours_in
    #For all the mean columns fill data for NA with following rule until NA is converted to value
    # (1 - Fill Forward, if there is data in earlier hour move to later hour. 2 - Take mean for that GROUP BY. 3 - Put it as 0)
    out.loc[:,idx[:,'mean']] = out.loc[:,idx[:,'mean']].groupby(GRP_BY_COL).fillna(method='ffill').groupby(GRP_BY_COL).fillna(icustay_means).fillna(0)
    return out 
  
def mean(a):
    return float(sum(a) / len(a))

def convert_txt_to_entity(text):
    doc = med7(text)
    result = ([(ent.text, ent.label_) for ent in doc.ents])
    return result

def convert_ent_to_embedding(ent_arr):
    out = list()
    try:
        if len(ent_arr) != 0:
            for v, k in ent_arr:
                v = v.lower().strip()
                if len(v.split(" ")) > 1:
                    vec = []
                    for each_word in v.split(" "):
                        if each_word in w2vec.wv.key_to_index:
                            vec.append(w2vec.wv[each_word])
                    if (len(vec) >0):
                        vec = [mean(x) for x in zip(*vec)]
                        out.append(vec)
                else:
                    if v in w2vec.wv.key_to_index:
                        out.append(w2vec.wv[v].tolist())
    except Exception as e:
        print(e)
    return out

def get_embd_array(df, embd_size = 64):
  p_ids = []
  overall_array = []
  for row in df.itertuples():
      data_item = []
      p_ids.append(row.SUBJECT_ID)
      for vec in row.embd:
        for embd in vec:
          data_item.append(embd)
          if(len(data_item) >= embd_size):
            break
        if(len(data_item) >= embd_size):
          break
      gap = embd_size - len(data_item)
      for d in range(gap):
        input = np.zeros(100).tolist()
        data_item.append(input)
      overall_array.append(data_item)
  return p_ids, overall_array
