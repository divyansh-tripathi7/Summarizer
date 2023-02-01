from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en import English
import numpy as np

nlp = English()
nlp.add_pipe('sentencizer')

corpus = ''' " The foundation of independence laid before 150 years in 1857 , which is described As First war of Independence got success after 100 long years of struggle & On 15th August 1947 when a new Democratic country which has taken over the attention of the world with it's principles of Non-violence & peace & above all had many challenges before it to stand in the world scenario. Today we have achieved a milestone by completing 60 years of independence. It's now the time for everyone or every Indian to undergo self-introspection of the achievements we already made & also those that are to be still achieved. While talking of achievements we should have a look at the challenges that India had at it's birth. We had problems of Social, economical, political etc . Partition of the country had serious consequences in the entire country. Country was experiencing violence, communal riots & a chaotic situation all over.First of all it needed a major attention to restore peace in the country to do anything else. On the economic front Britishers had exploited maximum resources of our country which is well explained by Dada Bhai Naoroji in his book titled "drain of wealth". Politically also we had to face many hurdles as we had no constitution , law of our own . The other most important thing was to establish a Democratic set up of Government in the country which at that time was felt almost impossible with diversified nature of India. So the country's position was no less than a baby learning walking ,applying all trial & error methods.'''

corpus3 = ''' The English have the reputation of being a nation of tea drinkers, but this wasn’t always the case. By the end of the 17th century, the English were the biggest coffee drinkers in the Western world, and coffee houses became the places to be seen. For gossip also, one could pick up talk of the latest intellectual developments in the field of science, politics, and so on, in this age of scientific discovery and research. Coffee houses were very simple and basic at first; one can say a room with a bar at one corner and a few plain tables and chairs at the other end. Customers paid a penny for a bowl – not a cup – of coffee. At that time, it was thought that the customers didn’t use bad language just because of the presence of a polite young woman. An added attraction was that coffee houses provided free newspapers and journals.

But people didn’t go to the coffee houses just to drink coffee. They went to talk. Simple cafes were converted and developed into clubs, where one with a penny could go for a drink and a chat. Most of them started to go to coffee houses to find other people with the same job or of same interest to talk and conduct business.

The great popularity of coffee houses lasted about a 100 years. In the later 18th century, increased trade with other countries made such luxuries as coffee cheaper and more easily available to the ordinary person. As a result, people started to drink it at home. At that time more tea was imported from abroad. The domestic tea-party replaced the century of the coffee house as the typical English social occasion.

 '''
 
 doc = nlp(corpus3.replace("\n", ""))

doc = nlp(corpus3)
sentences = []
for sent in doc.sents:
  sentences.append(sent)

len(sentences)
sentences.extend
sentences = [str(x) for x in sentences]
sentence_organizer = {k:v  for v , k in enumerate(sentences)}

tf_idf_vectorizer = TfidfVectorizer(min_df=2,  max_features=None, 
                                    strip_accents='unicode', 
                                    analyzer='word',
                                    token_pattern=r'\w{1,}',
                                    ngram_range=(1, 2), 
                                    use_idf=1,smooth_idf=1,
                                    sublinear_tf=1,
                                    stop_words = 'english')
                                    
                                    
tf_idf_vec = TfidfVectorizer(ngram_range=(1,3), smooth_idf = True, lowercase = False)
tf_idf_vec.fit(sentences)
sentence_vectors = tf_idf_vec.transform(sentences)

sentence_scores = np.array(sentence_vectors.sum(axis=1)).ravel()

N = 7
top_n_sentences = [sentences[ind] for ind in np.argsort(sentence_scores, axis=0)[::-1][:N]]


mapped_top_n_sentences = [(sentence,sentence_organizer[sentence]) for sentence in top_n_sentences]
print("Our top_n_sentence with their index: \n")
for element in mapped_top_n_sentences:
    print(element)

# Ordering our top-n sentences in their original ordering
mapped_top_n_sentences = sorted(mapped_top_n_sentences, key = lambda x: x[1])
ordered_scored_sentences = [element[0] for element in mapped_top_n_sentences]

# Our final summary
summary = " ".join(ordered_scored_sentences)

print(summary)
