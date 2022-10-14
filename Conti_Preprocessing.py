# CONTI PREPROCESSING 

df = pd.read_json("chatperActors.json")

# from funcy.flow import post_processing
cols = list(df.columns.values)
df[cols[1]] = df[cols[1]].apply(str)
post = df[cols[1]]

stop_words = stopwords.words("english")
stop_words.extend(['from', 'go', 'will', 'hello', 'place', 'bro', 'place', "fuck"])

def clean(data):
  clean = []
  for text in post:
    text = re.sub(r'http\S+', '', text)
    text = text.lower()
    text = re.sub("[^a-zA-Z']", ' ', text)
    text = re.sub("[\s+]", ' ', text) 
    clean.append(text)
  return clean 

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

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

data_words = remove_stopwords(data_words)


#BIGRAMS AND TRIGRAMS
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=50) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=50)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

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


for i in range(len(data_bigrams_trigrams)):
  print(data_bigrams_trigrams[i])
  if len(data_bigrams_trigrams[i])< 3 :
    data_bigrams_trigrams.pop(i)

acteurs = []

for i in range(len(df[0])):
  if df[0][i] not in acteurs:
    acteurs.append(df[0][i])

test = data_bigrams_trigrams

for i in range(len(test)):
  if len(test[i])< 100:
    acteurs.pop(i)
    test.pop(i)

#check if both are the same lenght
print(len(acteurs))
print(len(test))    

removed = []
X = ['many', 'andy', 'ghost', 'defender', 'stern', 'love', 'grant', 'zulas', 'pin', 'reshaev',
 'target', 'van', 'green', 'kaktus', 'mors', 'baget', 'bentley', 'mentos', 'price', 'steller', 
 'zloysobaka', 'idgo', 'azot', 'bill', 'hof', 'talar', 'buza', 'mushroom', 'bourbon', 'axel', 
 'boby', 'carter', 'mango', 'terry', 'logan', 'dick', 'kerasid', 'naned', 'hash', 'fast', 
 'mavelek', 'marsel', 'frog', 'salamandra', 'sunday', 'tiktak', 'braun', 'max', 'troy', 
 'professor', 'cany', 'strix', 'dandis', 'tunri', 'buri', 'tilar', 'total', 'electronic', 
 'noman', 'rozetka', 'mavemat', 'poll', 'ali', 'nevada', 'm', 'kevin', 'stakan', 'atlant',
  'rand', 'skywalker', 'bullet', 'derek', 'elon', 'viper', 'chip', 'cosmos', 'flip', 'bonen', 
  'dorirus', 'shell', 'urbanone', 'revers', 'clickclack', 'globus', 'cybergangster', 'chain', 
  'best', 'modnik', 'bio', 'pumba', 'ford', 'bloodrush', 'veron', 'skippy', 'sticks', 'netwalker', 
  'dominik', 'cheesecake', 'hors', 'grom', 'driver', 'gold', 'ramon', 'dove', 'gorec', 'dino', 'atlas', 
  'tom', 'ttrr@conference.q3mcco35auwcstmt.onion', 'specter', 'dollar', 'buran', 'elvira', 'wind', 
  'derekson', 'lemur', 'begemot', 'demon', 'tramp', 'fox', 'darc', 'taker', 'grand', 'revan', 'leo',
   'wertu', 'neo', 'gus', 'admintest', 'twin', 'vampire', 'spoon', 'bekeeper', 'spider', 'mont', 'ceram', 'casper']
for i in range(len(acteurs)):
  if acteurs[i] not in X:
    removed.append(acteurs[i])

#TF-IDF REMOVAL 
from gensim.models import TfidfModel

id2word = corpora.Dictionary(test)
id2word.filter_extremes(no_below=10, no_above=0.5)
print('Total Vocabulary Size:', len(id2word))

texts = test
corpus = [id2word.doc2bow(text) for text in texts]

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
