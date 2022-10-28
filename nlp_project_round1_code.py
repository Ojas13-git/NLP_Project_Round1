# -*- coding: utf-8 -*-
"""NLP_Project_Round1_code.ipynb



import nltk
import re

"""# Task-1: Import Two Books T1 and T2

Here we have imported two books from (https://www.gutenberg.org/ebooks/11) and  (https://www.gutenberg.org/ebooks/46) as opened as T1 and T2 using the open function of the OS library.

"""

filename="C:\\Users\\sourabh agarwal\\Desktop\\Alice.txt"
f = open(str(filename), 'r', encoding="utf8")

T1 = f.read()
f.close()

filename="C:\\Users\\sourabh agarwal\\Desktop\\Carol.txt"
f = open(str(filename), 'r', encoding="utf8")

T2 = f.read()
f.close()

"""# Task-2 : Text pre-processing and Tokenization

Text preprocessing:
First of all, we have cleaned the chapters name and all the other stuff which is not required for further processing.
Here, we have seen a pattern that after every 4 new empty lines we see chapter number and in the next line we see chapter name and two more new empty lines, which we removed using the REGEX, for both books T1 and T2 and called the now obtained books as T1_Clean and T2_clean.

"""

T1 = T1[T1.find("CHAPTER XII.")+len("CHAPTER XII.   Alice’s Evidence"):T1.find("*** END OF THE PROJECT GUTENBERG EBOOK")]
T1_clean = re.sub(r'\n{4}.*\n.*\n\n\n+', '\n\n', T1)
# print(T1_clean)

T2 = T2[T2.find("Stave   V")+len("Stave   V: The End of It"):T2.find("*** END OF THE PROJECT GUTENBERG EBOOK")]
T2_clean = re.sub(r'\n{3}.*\n\n+', '\n\n', T2)
# print(T2_clean)

"""Tokenization: Then the next step was to tokenize both the books, for that we have imported word_tokenizer from nltk.tokenize. Then we one by one we have tokenized both the cleaned books and stored the resultant (tokenized) list in Token_T1 and Token_T2 for book T1 and T2 respectively."""

from nltk.tokenize import word_tokenize

Token_T1=word_tokenize(T1_clean)
# Token_T1

from nltk.tokenize import word_tokenize

Token_T2=word_tokenize(T2_clean)
# Token_T2

"""# Task-3 : Freq Distribution Of T1 and T2

Here in this step before finding the distribution, we have removed all the punctuation and the numerical values, i.e. everything except the alphabetical words, so that they don’t pop in the frequency table/curve, and make our findings wrong.
"""

import re
punctuation = re.compile(r"[^a-zA-Z]+")
T1_only_words = []
for words in Token_T1:
  words = punctuation.sub("", words)
  if len(words) > 0:
    T1_only_words.append(words.lower())
# print(T1_only_words)
Token_T1 = T1_only_words

import re
punctuation = re.compile(r"[^a-zA-Z]+")
T2_only_words = []
for words in Token_T2:
  words = punctuation.sub("", words)
  if len(words) > 0:
    T2_only_words.append(words.lower())
# print(T2_only_words)
Token_T2 = T2_only_words

"""
Then we have imported the FreqDist function. And took the count for frequency for each word. We did so for both Token_T1 and Token_T2. Then we took the 20 most common words for both books, using most_common() available in FreqDist through nltk.probability.
"""

from nltk.probability import FreqDist

fdist_T1=FreqDist()
for word in Token_T1:
    fdist_T1[word]+=1
fdist_T1

fdist_T2=FreqDist()
for word in Token_T2:
    fdist_T2[word]+=1
fdist_T2

fdist_T1_top20 = fdist_T1.most_common(20)
fdist_T1_top20

fdist_T2_top20 = fdist_T2.most_common(20)
fdist_T2_top20

"""
Here we have plotted for the 20 most common words for both the books, and clearly word ‘the’ is the most common word in both the books, but that’s could be deceptive as `the` is a stopword, which will be taken care in next steps, and then we’ll plot again and will see how the plot changes."""

fdist_T1.plot(20, cumulative=False)

fdist_T2.plot(20, cumulative=False)

"""# Task-4: Word Cloud Of T1 and T2

Now to show the wordcloud from frequency distribution we have used the wordcloud library along with the Matplotlib library.
As we can see here word `the`, which is most frequent so far is plotted the largest and other were also being plotted according to their frequency (before removing the stopwords).

"""

from wordcloud import WordCloud
import matplotlib.pyplot as plt

word_cloud_dict1 = fdist_T1
wordcloud1 = WordCloud(width = 500, height = 300).generate_from_frequencies(word_cloud_dict1)
plt.figure(figsize=(15,8))
plt.imshow(wordcloud1)
plt.axis("off")
plt.show()

word_cloud_dict2 = fdist_T2
wordcloud2 = WordCloud(width = 500, height = 300).generate_from_frequencies(word_cloud_dict2)
plt.figure(figsize=(15,8))
plt.imshow(wordcloud2)
plt.axis("off")
plt.show()

"""# Task-5: Removal of Stop Words And Creating Word Cloud

Now, this is the time when we are separating stopwords from our list of tokens for both books. Here firstly we have found the stopwords for the English language. Then one by one, we examined all the tokens and if that token is not a stopword, we considered it for further processing and will store them in a new list, and the words left will be of no use and hence won’t include them in the new list. This way we have removed the stopwords from Token_T1 and Token_T2. Then we have again found the frequency distribution for the cleaned set of tokens and plotted the new word graphs. As we can see stopwords like ‘a’, ‘i’, ‘the’ are no more appear in the frequency distribution as well as in the word cloud. And thus the most frequent word for Book1 i.e. Alice’s adventures in Wonderland is ‘said’ and for Book2 i.e.  A Christmas Carol in Prose; Being a Ghost Story of Christmas is ‘scrooge’.
"""

from nltk.corpus import stopwords
sr= stopwords.words('english')
clean_tokens_T1 = []

for token in Token_T1:
    if token not in stopwords.words('english'):
        clean_tokens_T1.append(token)

freq_T1 = nltk.FreqDist(clean_tokens_T1)

#for key,val in freq_T1.items():
   # print(str(key) + ':' + str(val))

from nltk.corpus import stopwords
sr= stopwords.words('english')
clean_tokens_T2 = []

for token in Token_T2:
    if token not in stopwords.words('english'):
        clean_tokens_T2.append(token)

freq_T2 = nltk.FreqDist(clean_tokens_T2)
# for key,val in freq_T2.items():
   # print(str(key) + ':' + str(val))

word_cloud_dict3 = freq_T1
wordcloud3 = WordCloud(width = 500, height = 300).generate_from_frequencies(word_cloud_dict3)
plt.figure(figsize=(15,8))
plt.imshow(wordcloud3)
plt.axis("off")
plt.show()

word_cloud_dict4 = freq_T2
wordcloud4 = WordCloud(width = 1000, height = 500).generate_from_frequencies(word_cloud_dict4)
plt.figure(figsize=(15,8))
plt.imshow(wordcloud4)
plt.axis("off")
plt.show()

"""# Task-6: Relationship Between Word Length And Frequency

Next, we are going to find the frequency distribution with the count. For that first, we have found various lengths of words for each cleaned set of words of each book. Then for each possible word size, we found the number of words with that size and then stored all the pair of word length and number of such length words and then converted it to the data frame, which we further plotted using the matplotlib library, and visualized the relationship between length and frequency for both cleaned token set of book T1 and T2.
"""

import pandas as pd

data = []

fdist_len_T1 = FreqDist()
for word in clean_tokens_T1:
    fdist_len_T1[len(word)] += 1

for key in sorted(fdist_len_T1):
    temp = []
    temp.append(key)
    temp.append(fdist_len_T1[key])
    data.append(temp)
    
df_T1 = pd.DataFrame(data, columns = ['Length', 'Count'])
print(df_T1)

import pandas as pd

data = []

fdist_len_T2 = FreqDist()
for word in clean_tokens_T2:
    fdist_len_T2[len(word)] += 1

for key in sorted(fdist_len_T2):
    temp = []
    temp.append(key)
    temp.append(fdist_len_T2[key])
    data.append(temp)

df_T2 = pd.DataFrame(data, columns = ['Length', 'Count'])
print(df_T2)

import matplotlib.pyplot as plt
plt.plot(df_T1['Length'],df_T1['Count'])
plt.plot(df_T2['Length'],df_T2['Count'])
plt.xlabel('Length of word')
# naming the y axis
plt.ylabel('Number of Word')
 
# giving a title to My graph
plt.title('Word Length and Frequency Comaprison for Both Garphs')
plt.show()

"""# Task-7 : POS Tagging For T1 and T2

In this step, we have done the part of speech tagging for the words of each book. For this, we have used the nltk.pos_tag function which uses the Penn Treebank tags. Then to get the distribution of various tags we have used the Counter function and then and then printed the tag wise distribution in descending order, for both the token sets T1 and T2.
"""

Token1_pos = nltk.pos_tag(clean_tokens_T1)
# Token1_pos

Token1_pos = list(map(list, Token1_pos))
# Token1_pos

Token2_pos = nltk.pos_tag(clean_tokens_T2)
# Token2_pos

Token2_pos = list(map(list, Token2_pos))
# Token2_pos

from collections import Counter
pos_tag_T1 = Counter(tag for _, tag in Token1_pos)

for k, v in sorted(pos_tag_T1.items(), key=lambda kv: kv[1], reverse=True):
    print(f"{k}: {v}")

from collections import Counter
pos_tag_T2 = Counter(tag for _, tag in Token2_pos)

for k, v in sorted(pos_tag_T2.items(), key=lambda kv: kv[1], reverse=True):
    print(f"{k}: {v}")