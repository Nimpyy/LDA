import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
import re
import unidecode
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from wordcloud.wordcloud import STOPWORDS
from wordcloud import WordCloud
import numpy as np
import copy
import pickle
import re
import math
from textblob import TextBlob as tb 
from tqdm import tqdm
import string
from spellchecker import SpellChecker
import collections

# df = pd.read_csv('exported_case_18_29314.csv', delimiter=",")
df = pd.read_csv("apres modified.csv", delimiter=";", encoding = 'ISO-8859-1')
cols = list(df.columns.values)
post = df['Content'].apply(str)
title = df['Title'].apply(str)
longueur_index = len(post)

list_index = []
for i in range(longueur_index):
  list_index.append(post[i])

list_title = []
for i in range(longueur_index):
  list_title.append(title[i])

# LEMATIZE FUNCTION 

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

def get_wordnet_pos(word):

    """Map POS tag to first character lemmatize() accepts"""
    # print(word)
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

nltk.download('averaged_perceptron_tagger')

def lematize(data):
  
  newtext = ""
  lemmatizer = WordNetLemmatizer()
  tokens = word_tokenize(data)

  newtext = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in tokens]
  return newtext


def clean(text, langage_filtre):
    # sentences = []

    text = re.sub(r'http\S+', '', text)
        # get rid of links
        
    #on enlève les accents
    # phrase = unidecode.unidecode(phrase)                //pour le francais
    text = text.lower()
    # print(text)
        #remplace les caractères spéciaux par des espaces
    text = re.sub("[^a-zA-Z']", ' ', text)
    # print(text)
        #supprime les espaces inutiles
    text = re.sub("[\s+]", ' ', text)
    # print(text)    
        #tokenize
    

    TT = TweetTokenizer()
    mots = TT.tokenize(text)
    # print(mots) 

        #filtre les mots par stopWords + #garde les mots avec 2 lettres minimum + #lemmatise le mot
    new_text = ''
    stopWords = set(stopwords.words(langage_filtre))
    wnl = WordNetLemmatizer()
    for mot in mots:
        if mot not in stopWords:
            if len(mot) > 2:
                new_text = new_text + " " + mot
       
        #on ajoute la phrase cleaned
    new_text = lematize(new_text)
    new_text = " ".join(new_text)
   
    # sentences.append(new_text)
    return new_text
    # return sentences

clean_post = []

for i in tqdm(range(len(list_index))):
  clean_post.append(clean(list_index[i], 'english'))

clean_title = []

for i in tqdm(range(len(list_title))):
  clean_title.append(clean(list_title[i], 'english'))



