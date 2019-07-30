# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 14:02:36 2018

@author: SusarlaS
"""

import nltk
import os
import platform
import sys
import re
import pandas as pd
from sympound import sympound
from textblob import TextBlob

distancefun = None
if platform.system() != "Windows":
    from pyxdameraulevenshtein import damerau_levenshtein_distance
    distancefun = damerau_levenshtein_distance
else:
    from jellyfish import levenshtein_distance
    distancefun = levenshtein_distance


ssc = sympound(distancefun=distancefun, maxDictionaryEditDistance=10)
ssc.load_dictionary('C:/Users/susarlas/Desktop/ml/data/DICTS/ADPtokens.txt' )


#sys.path.insert(0,'C:/Users/susarlas/Desktop/ml/project33')
input_dir = "C:/Users/susarlas/Desktop/test/"


for dirpath, dirs, files in os.walk(input_dir):	
	for filename in files:
         fname = os.path.join(dirpath,filename)
         in_file = open(fname,"rt+",encoding='utf-8')
         input_text = in_file.read()         # read the entire file into a string variable
         input_tempo = input_text    
         input_tempo = input_tempo.strip()
         input_tempo = input_text.upper()
         input_dummy = input_tempo
         nltk_tokens = nltk.word_tokenize(input_dummy)
         
         for nl_tk in range(len(nltk_tokens)):
             tkn = nltk_tokens[nl_tk]
             print('actual'+tkn)
             a = str(ssc.lookup_compound(input_string = tkn,edit_distance_max = 10))
             rev = a[::-1]
             rev_int = int(rev[0])
             if rev_int <= 1:
                b = a.split(':')[0]  
                print('a'+a)
                print('b'+b)
#                ssc.save_pickle("symspell.pickle")
#                ssc.load_pickle("symspell.pickle")
#                print('before modification'+nltk_tokens[nl_tk])
                nltk_tokens[nl_tk] = b
                print('after modification'+nltk_tokens[nl_tk])
         data = nltk_tokens    
################# DETOKENIZE ##############################################
    
         dfd = ' '.join(data) 
         input_text = ''
#         text = TextBlob(input_dummy)
#         dfd = str(text.correct())
         input_text = dfd
         print('after correction')
         input_text = input_text.upper()
         in_file.seek(0)
         in_file.write(input_text)
         in_file.truncate()
         in_file.close() 
         
  
    

