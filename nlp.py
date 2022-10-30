import re
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

#Task1
file="T1.txt"
f = open(str(file), 'r', encoding="utf8")
T1 = f.read()
f.close()


#Task2-tokenize and pre-prossesing(punctuation)
Token_T1=word_tokenize(T1)
Token_T1
print(Token_T1)


punctuation = re.compile(r"[^a-zA-Z]+")
T1_revised = []
for words in Token_T1:
  words = punctuation.sub("", words)
  if len(words) > 0:
    T1_revised.append(words.lower())
Token_T1 = T1_revised


##task3-frequency distribution
freq_T1=FreqDist()
for word in Token_T1:
    freq_T1[word]+=1
freq_T1

#for key,val in freq_T1.items():
    #print(str(key) + ':' + str(val))

freq_T1.plot(20, cumulative=False)

##task4-word cloud
word_cloud_dict1 = freq_T1
wordcloud1 = WordCloud(width = 500, height = 300).generate_from_frequencies(word_cloud_dict1)
plt.figure(figsize=(20,10))
plt.imshow(wordcloud1)
plt.axis("off")
plt.show()

##task5--removing stop words and again plotting word cloud
from nltk.corpus import stopwords
sr= stopwords.words('english')
T1_new = []

for token in Token_T1:
    if token not in stopwords.words('english'):
        T1_new.append(token)

freq_T1 = nltk.FreqDist(T1_new)

#for key,val in freq_T1.items():
    #print(str(key) + ':' + str(val))


word_cloud_dict2 = freq_T1
wordcloud2 = WordCloud(width = 500, height = 300).generate_from_frequencies(word_cloud_dict2)
plt.figure(figsize=(20,10))
plt.imshow(wordcloud2)
plt.axis("off")
plt.show()


##task6- relationship b/w word length & freq
data = []
fdist_len_T1 = FreqDist()
for word in  T1_new:
    fdist_len_T1[len(word)] += 1

for key in sorted(fdist_len_T1):
    temp = []
    temp.append(key)
    temp.append(fdist_len_T1[key])
    data.append(temp)

df_T1 = pd.DataFrame(data, columns=['Length', 'Count'])
print(df_T1)

plt.plot(df_T1['Length'],df_T1['Count'])
plt.xlabel('Length of word')
plt.ylabel('Number of Word')
plt.title('Word Length and Frequency Comparision')
plt.show()


##task 7 -pos tagging
Token1_pos = nltk.pos_tag(T1_new)
Token1_pos = list(map(list, Token1_pos))

from collections import Counter
pos_tag_T1 = Counter(tag for _, tag in Token1_pos)

for k, v in sorted(pos_tag_T1.items(), key=lambda kv: kv[1], reverse=True):
    print(f"{k}: {v}")
