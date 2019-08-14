
import nltk
import random
from nltk.corpus import movie_reviews

#Get all words in the movie reviews
words = []
for w in movie_reviews.words():
    words.append(w.lower())

#Get frequencies of all words
words = nltk.FreqDist(words)


#There are two categories pos,neg
#for every category, get the files associated with it.
# Add the review words, the category as a tuple to the reviews
reviews = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

#Shuffle the data since arranged categorywise
random.shuffle(reviews)







