pip install pyspellchecker
import re
# import unidecode
import nltk 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

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
        # if mot not in stopWords:
            # if len(mot) > 2:
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


TT = TweetTokenizer()
post = []
title = []

# for i in tqdm(clean_post[0]):
post = TT.tokenize(str(clean_post))
post = [''.join(c for c in s if c not in string.punctuation) for s in post]
post = [s for s in post if s]

title = TT.tokenize(str(clean_title))
title = [''.join(c for c in s if c not in string.punctuation) for s in title]
title = [s for s in title if s]

 # need to clean data before finding it to that ( remove punctuation, to lowercase but don't remove stopwords)

spell = SpellChecker(distance=2)
spell = SpellChecker()

# find those words that may be misspelled
misspelled = list(spell.unknown(title))

print("Possible list of misspelled words in the original text:\n",misspelled)
print(len(misspelled))

blank = []

for i in tqdm(range(len(title))):
  if title[i] not in misspelled :
    blank.append(title[i])

print(blank[:150])
print(len(blank))

errors =[]
for i in tqdm(range(len(title))):
  if title[i] not in blank:
    errors.append(title[i])

print(len(errors))
print(errors[0:100])

occurrences = collections.Counter(errors)
df = pd.DataFrame.from_dict(occurrences, orient='index').reset_index()