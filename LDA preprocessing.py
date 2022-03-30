pip install pyLDAvis
import numpy as np
import json
import glob
import pandas as pd
import regex as re

#Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

#spacy
import spacy
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

#vis
import pyLDAvis
import pyLDAvis.gensim_models
from gensim.models import TfidfModel

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

df = pd.read_csv("apres modified.csv", delimiter=";")

cols = list(df.columns.values)
print(df["Content"])

df['Content']=df['Content'].apply(str)
df['Ttitle']=df['Title'].apply(str)

post = df['Content']
title = df['Title']

stop_words = stopwords.words("english")
stop_words.extend(['from', 'go', 'will'])
print(stop_words)

def clean(data):
  clean = []
  for text in post:
    text = re.sub(r'http\S+', '', text)
    text = text.lower()
    text = re.sub("[^a-zA-Z']", ' ', text)
    text = re.sub("[\s+]", ' ', text) 
    clean.append(text)
  return clean 

print(clean(post)[2214][0:150])

def lemmatization(texts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
  nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
  texts_out = []
  for text in texts:
    doc = nlp(text)
    new_text = []
    for token in doc:
      if token.pos_ in allowed_postags:
        new_text.append(token.lemma_)
    final = " ".join(new_text)
    texts_out.append(final)
  return (texts_out)

lemmatized_texts = lemmatization(clean(post))
# lemmatized_texts = lemmatization(post)
print(lemmatized_texts[0][0:20])

def gen_words(texts):
    final = []
    for text in texts:
        new = gensim.utils.simple_preprocess(text, deacc=True)
        final.append(new)
    return (final)

data_words = gen_words(lemmatized_texts)

print (data_words[0][0:150])

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

data_words = remove_stopwords(data_words)


#BIGRAMS AND TRIGRAMS
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=50) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=50)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
print(trigram_mod[bigram_mod[data_words[0]]])

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

    # Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

data_bigrams_trigrams = make_trigrams(data_words_bigrams)

print(data_bigrams_trigrams[0:5])

# tf-idf removal 

id2word = corpora.Dictionary(data_bigrams_trigrams)

id2word.filter_extremes(no_below=10, no_above=0.5)
print('Total Vocabulary Size:', len(id2word))

texts = data_bigrams_trigrams

corpus = [id2word.doc2bow(text) for text in texts]

print(corpus[0][0:25])

tfidf = TfidfModel(corpus, id2word=id2word)

low_value = 0.03
words = []
words_missing_in_tfidf = []
for i in range(0, len(corpus)):
  bow = corpus[i]
  low_value_words = []
  tfidf_ids = [id for id, value in tfidf[bow]]
  bow_ids = [id for id, value in bow]
  low_value_words = [id for id, value in tfidf[bow] if value < low_value]
  drops = low_value_words+words_missing_in_tfidf
  for item in drops:
    words.append(id2word[item])
  words_missing_in_tfidf = [id for id in bow_ids if id not in tfidf_ids]

  new_bow = [b for b in bow if b[0] not in low_value_words and b[0] not in words_missing_in_tfidf]
  corpus[i] = new_bow

