import re
import nltk
#nltk.download('omw-1.4')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('maxent_ne_chunker')
#nltk.download('words')
from nltk.corpus import wordnet
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from wordcloud import WordCloud
from nltk import ne_chunk,pos_tag
from nltk.sem.relextract import extract_rels,rtuple


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



#############################                          Round02                          #############################

##part 01
is_noun = lambda pos: pos[:2] == 'NN'
nounsBook1 = [word for (word, pos) in Token1_pos if is_noun(pos)]
nounsBook1

is_verb = lambda pos: pos[:3] == 'VBG'
verbsBook1 = [word for (word, pos) in Token1_pos if is_verb(pos)]
verbsBook1



parentBook1_noun = []
for noun in nounsBook1:
    if len(wordnet.synsets(noun)) > 0:
        x = wordnet.synsets(noun)[0]
        if len(x.hypernyms()) > 0:
            y = [noun, x.hypernyms()[0]]
            parentBook1_noun.append(y)
        print(f'{noun} -> {x.hypernyms()}')


parentBook1_verb = []
for noun in verbsBook1:
    if len(wordnet.synsets(noun)) > 0:
        x = wordnet.synsets(noun)[0]
        if len(x.hypernyms()) > 0:
            y = [noun, x.hypernyms()[0]]
            parentBook1_verb.append(y)
        print(f'{noun} -> {x.hypernyms()}')


freqBook1_noun = Counter(parent for word, parent in parentBook1_noun)
val = []
par = []
for k, v in sorted(freqBook1_noun.items(), key=lambda kv: kv[1], reverse=True):
    val.append(v)
    par.append(k.lemmas()[0].name())

plt.hist(val)
plt.show()

plt.figure(figsize = (20,8))
plt.bar(par[:100], val[:100])
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize = (20,8))
plt.bar(par[100:200], val[100:200])
plt.xticks(rotation=90)
plt.show()


freqBook1_verb = Counter(parent for word, parent in parentBook1_verb)
val = []
par = []
for k, v in sorted(freqBook1_verb.items(), key=lambda kv: kv[1], reverse=True):
    val.append(v)
    par.append(k.lemmas()[0].name())

plt.hist(val)
plt.show()

plt.figure(figsize = (20,8))
plt.bar(par[:100], val[:100])
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize = (20,8))
plt.bar(par[100:200], val[100:200])
plt.xticks(rotation=90)
plt.show()



##part02
NER_Book1 = set()
for sent in nltk.sent_tokenize(T1):
  for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
     if hasattr(chunk, 'label'):
       NER_Book1.add((' '.join(c[0] for c in chunk), chunk.label()))
NER_Book1
print(NER_Book1)

passage="""Artificial Intelligence (AI) is a big field, and this is a big book. We have tried to explore the
full breadth of the field, which encompasses logic, probability, and continuous mathematics;
perception, reasoning, learning, and action; and everything from microelectronic devices to
robotic planetary explorers. The book is also big because we go into some depth.
The subtitle of this book is “A Modern Approach.” The intended meaning of this rather
empty phrase is that we have tried to synthesize what is now known into a common framework, rather than trying to explain each subfield of AI in its own historical context. We
apologize to those whose subfields are, as a result, less recognizable.

New to this edition
This edition captures the changes in AI that have taken place since the last edition in 2003.
There have been important applications of AI technology, such as the widespread deployment of practical speech recognition, 
machine translation, autonomous vehicles, and household robotics. There have been algorithmic landmarks, such as the solution of the game of
checkers. And there has been a great deal of theoretical progress, particularly in areas such
as probabilistic reasoning, machine learning, and computer vision. Most important from our
point of view is the continued evolution in how we think about the field, and thus how we
organize the book. The major changes are as follows:
• We place more emphasis on partially observable and nondeterministic environments,
especially in the nonprobabilistic settings of search and planning. The concepts of
belief state (a set of possible worlds) and state estimation (maintaining the belief state)
are introduced in these settings; later in the book, we add probabilities.
• In addition to discussing the types of environments and types of agents, we now cover
in more depth the types of representations that an agent can use. We distinguish among
atomic representations (in which each state of the world is treated as a black box),
factored representations (in which a state is a set of attribute/value pairs), and structured
representations (in which the world consists of objects and relations between them).
• Our coverage of planning goes into more depth on contingent planning in partially
observable environments and includes a new approach to hierarchical planning.
• We have added new material on first-order probabilistic models, including open-universe
models for cases where there is uncertainty as to what objects exist.
• We have completely rewritten the introductory machine-learning chapter, stressing a
wider variety of more modern learning algorithms and placing them on a firmer theoretical footing.
• We have expanded coverage of Web search and information extraction, and of techniques for learning from very large data sets.
• 20% of the citations in this edition are to works published after 2003.
• We estimate that about 20% of the material is brand new. The remaining 80% reflects
older work but has been largely rewritten to present a more unified picture of the field.

The main unifying theme is the idea of an intelligent agent. We define AI as the study of
agents that receive percepts from the environment and perform actions. Each such agent implements a function that maps percept sequences to actions, and we cover different ways to
represent these functions, such as reactive agents, real-time planners, and decision-theoretic
systems. We explain the role of learning as extending the reach of the designer into unknown
environments, and we show how that role constrains agent design, favoring explicit knowledge representation and reasoning. We treat robotics and vision not as independently defined
problems, but as occurring in the service of achieving goals. We stress the importance of the
task environment in determining the appropriate agent design.
Our primary aim is to convey the ideas that have emerged over the past fifty years of AI
research and the past two millennia of related work. We have tried to avoid excessive formality in the presentation of these ideas while retaining precision. We have included pseudocode
algorithms to make the key ideas concrete; our pseudocode is described in Appendix B.
This book is primarily intended for use in an undergraduate course or course sequence.
The book has 27 chapters, each requiring about a week’s worth of lectures, so working
through the whole book requires a two-semester sequence. A one-semester course can use
selected chapters to suit the interests of the instructor and students. The book can also be
used in a graduate-level course (perhaps with the addition of some of the primary sources
suggested in the bibliographical notes). Sample syllabi are available at the book’s Web site,
aima.cs.berkeley.edu. The only prerequisite is familiarity with basic concepts of
computer science (algorithms, data structures, complexity) at a sophomore level. Freshman
calculus and linear algebra are useful for some of the topics; the required mathematical background is supplied in Appendix A.
Exercises are given at the end of each chapter. Exercises requiring significant programming are marked with a keyboard icon. These exercises can best be solved by taking
advantage of the code repository at aima.cs.berkeley.edu. Some of them are large
enough to be considered term projects. A number of exercises require some investigation of
the literature; these are marked with a book icon.
Throughout the book, important points are marked with a pointing icon. We have included an extensive index of around 6,000 items to make it easy to find things in the book.
Wherever a new term is first defined, it is also marked in the margin.

This book would not have been possible without the many contributors whose names did not
make it to the cover. Jitendra Malik and David Forsyth wrote Chapter 24 (computer vision)
and Sebastian Thrun wrote Chapter 25 (robotics). Vibhu Mittal wrote part of Chapter 22
(natural language). Nick Hay, Mehran Sahami, and Ernest Davis wrote some of the exercises.
Zoran Duric (George Mason), Thomas C. Henderson (Utah), Leon Reznik (RIT), Michael
Gourley (Central Oklahoma) and Ernest Davis (NYU) reviewed the manuscript and made
helpful suggestions. We thank Ernie Davis in particular for his tireless ability to read multiple
drafts and help improve the book. Nick Hay whipped the bibliography into shape and on
deadline stayed up to 5:30 AM writing code to make the book better. Jon Barron formatted
and improved the diagrams in this edition, while Tim Huang, Mark Paskin, and Cynthia
Bruyns helped with diagrams and algorithms in previous editions. Ravi Mohan and Ciaran
O’Reilly wrote and maintain the Java code examples on the Web site. John Canny wrote
the robotics chapter for the first edition and Douglas Edwards researched the historical notes.
Tracy Dunkelberger, Allison Michael, Scott Disanno, and Jane Bonnell at Pearson tried their
best to keep us on schedule and made many helpful suggestions. Most helpful of all has
been Julie Sussman, P. P. A ., who read every chapter and provided extensive improvements. In
previous editions we had proofreaders who would tell us when we left out a comma and said
which when we meant that; Julie told us when we left out a minus sign and said xi when we
meant xj . For every typo or confusing explanation that remains in the book, rest assured that
Julie has fixed at least five. She persevered even when a power failure forced her to work by
lantern light rather than LCD glow.

Stuart would like to thank his parents for their support and encouragement and his
wife, for her endless patience and boundless wisdom.RUGS (Russell’s Unusual Group of Students) have been unusually
helpful, as always.
Peter would like to thank his parents  for getting him started,
and his wife , children , colleagues, and friends for encouraging and
tolerating him through the long hours of writing and longer hours of rewriting.
We both thank the librarians at Berkeley, Stanford, and NASA and the developers of
CiteSeer, Wikipedia, and Google, who have revolutionized the way we do research.
"""

passageNER = set()
for sent in nltk.sent_tokenize(passage):
  for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
     if hasattr(chunk, 'label'):
       passageNER.add((' '.join(c[0] for c in chunk), chunk.label()))
passageNER

tp, tn, fp, fn = 0, 0, 0, 0

prediction = dict()
for x in passageNER:
    prediction.setdefault(x[0], []).append(x[1])
# print(prediction)

HandMarked= {
('Zoran', 'PERSON'),
    ('Thomas C. Henderson', 'PERSON'),
    ('Michael Gourley', 'PERSON'),
    ('Ernest Davis', 'PERSON'),
    ('Ernie Davis', 'PERSON'),
    ('Nick Hay', 'PERSON'),
    ('Stuart', 'PERSON'),
    ('Peter', 'PERSON'),
    ('George Mason', 'ORGANIZATION'),
    ('Utah', 'ORGANIZATION'),
    ('Central Oklahoma', 'ORGANIZATION'),
    ('Berkeley', 'ORGANIZATION'),
    ('NASA', 'ORGANIZATION'),
    ('Stanford', 'ORGANIZATION')

}

actual = dict()
for x in HandMarked:
    actual.setdefault(x[0], x[1])
# print(actual)


for entity, type_ in actual.items():
    # print(entity, ":", type_)
    if type_ in prediction[entity]:
        tp += 1
        fp += len(prediction[entity]) - 1
    else:
        fp += len(prediction[entity])
    if type_ not in prediction[entity]:
        fn += 1
    for x in ['GPE', 'ORGANIZATION', 'PERSON']:
        if x != type_ and x not in prediction[entity]:
            tn += 1

print("TP: ", tp, " TN: ", tn, " FP: ", fp, " FN: ", fn)
F1_score = (2 * tp) / ((2 * tp) + fp + fn)
accuracy = (tp + tn) / (tp + tn + fn + fp)
print("Accuracy: {0:.2f}".format(accuracy))
print("F1 Score: {0:.2f}".format(F1_score))


#part 03

##person-location relationship
BELONG = re.compile(r".*\bin|from|belong|lived\b.*")
sentences = nltk.sent_tokenize(T1)
tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
tagged_sentences =[nltk.pos_tag(sentence) for sentence in tokenized_sentences]

for i,sent in enumerate(tagged_sentences):
    sent = ne_chunk(sent)
    rels=extract_rels('PER','GPE',sent,corpus='ace',pattern = BELONG, window=10)
    for rel in rels:
        print(rtuple(rel))


##person-person relationship
RELATIONS = re.compile(r".*\bmother|father|sister|brother|aunt|uncle|wife|husband|colleague\b.*")

for i,sent in enumerate(tagged_sentences):
    sent = ne_chunk(sent)
    rels=extract_rels('PER','PER',sent,corpus='ace',pattern = RELATIONS, window=10)
    for rel in rels:
        print(rtuple(rel))


##person-organization relationship
ORG = re.compile(r".*\bwork|of|in\b.*")

for i,sent in enumerate(tagged_sentences):
    sent = ne_chunk(sent)
    rels=extract_rels('PER','ORG',sent,corpus='ace',pattern = ORG, window=10)
    for rel in rels:
        print(rtuple(rel))

