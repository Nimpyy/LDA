tm_results = optimal_model[corpus]
corpus_topics = [sorted(topics, key=lambda record: -record[1])[0] for topics in tm_results]        #We can get the most dominant topic of each document as below:
corpus_topics

for i in range(len(corpus)):
  print(optimal_model[corpus[i]])

topics = [[(term, round(wt, 3)) for term, wt in optimal_model.show_topic(n, topn=20)] for n in range(0, optimal_model.num_topics)]
topics_df = pd.DataFrame([[term for term, wt in topic] for topic in topics], columns = ['Term'+str(i) for i in range(1, 21)], index=['Topic '+str(t) for t in range(1, optimal_model.num_topics+1)]).T
topics_df.head()

pd.set_option('display.max_colwidth', -1)
topics_df = pd.DataFrame([', '.join([term for term, wt in topic]) for topic in topics], columns = ['Terms per Topic'], index=['Topic'+str(t) for t in range(1, optimal_model.num_topics+1)] )
topics_df

#TOPICS WORDCLOUD
import matplotlib.pyplot as plt
# import wordclouds
from wordcloud import WordCloud

# initiate wordcloud object
wc = WordCloud(background_color="white", colormap="Dark2", max_font_size=150, random_state=42)

# set the figure size
plt.rcParams['figure.figsize'] = [20, 15]

# Create subplots for each topic
for i in range(10):

    wc.generate(text=topics_df["Terms per Topic"][i])
    
    plt.subplot(5, 4, i+1)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(topics_df.index[i])

plt.show()

from gensim.models.ldamodel import LdaModel

def convertldaMalletToldaGen(mallet_model):
    model_gensim = LdaModel(
        id2word=mallet_model.id2word, num_topics=mallet_model.num_topics,
        alpha=mallet_model.alpha) 
    model_gensim.state.sstats[...] = mallet_model.wordtopics
    model_gensim.sync_state()
    return model_gensim

# !pip install pyLDAvis
import pyLDAvis 
import pyLDAvis.gensim_models as gensimvis
vis_data = gensimvis.prepare(optimal_model, corpus, id2word, sort_topics=False)
pyLDAvis.display(vis_data)

from matplotlib import pyplot as plt

names = ['Internal tasking and management', 'Technical talk & workforce ', 'HR activities', 'Malware ', 'Customer service & Problem-solving ']
values = [1.46, 16.79, 52.55, 7.3, 21.9]

plt.figure(figsize=(60, 10))

plt.subplot(131)
plt.bar(names, values)

plt.show()

# create a dataframe
corpus_topic_df = pd.DataFrame()
# get the Titles from the original dataframe
# corpus_topic_df['Title'] = df.Title
corpus_topic_df['Dominant Topic'] = [item[0]+1 for item in corpus_topics]
corpus_topic_df['Contribution %'] = [round(item[1]*100, 2) for item in corpus_topics]
corpus_topic_df['Topic Terms'] = [topics_df.iloc[t[0]]['Terms per Topic'] for t in corpus_topics]
corpus_topic_df.head()


dominant_topic_df = corpus_topic_df.groupby('Dominant Topic').agg(
                    Doc_Count = ('Dominant Topic', np.size),                                     
                    Total_Docs_Perc = ('Dominant Topic', np.size)).reset_index()                   

dominant_topic_df['Total_Docs_Perc'] = dominant_topic_df['Total_Docs_Perc'].apply(lambda row: round((row*100) / len(corpus), 2))
dominant_topic_df

#We can also get which document makes the highest contribution to each topic:
df = corpus_topic_df.groupby('Dominant Topic').apply(lambda topic_set: (topic_set.sort_values(by=['Contribution %'], ascending=False).iloc[0])).reset_index(drop=True)
df

#SAVE YOUR BEST MODEL
ldagensim.save("MODEL137acteurk5.model")