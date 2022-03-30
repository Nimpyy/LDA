from wordcloud.wordcloud import STOPWORDS
from wordcloud import WordCloud

text_P = " ".join(clean_post)
text_T = " ".join(clean_title)


# Create a WordCloud object
WC = WordCloud(width=500,height=500, max_words=200,
               background_color="white", max_font_size=120, 
               collocations = False, random_state=42, contour_width=3,
               contour_color='steelblue',relative_scaling=1, stopwords = STOPWORDS) 

WC.generate(text_T)
WC.to_image()

def getFreq(text):
  counts = WordCloud().process_text(text)

  return dict(sorted(counts.items(), key = lambda item: item[1], reverse = True))

df = pd.DataFrame(getFreq(text_T).items(), columns=['Word', 'Freq'])