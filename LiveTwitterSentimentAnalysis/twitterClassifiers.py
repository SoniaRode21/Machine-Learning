'''
The following code unpickles the trained classifiers.
__author__='Soniya Rode'
__citation__="pythonprogramming"
'''
import nltk
import random
from nltk.tokenize import word_tokenize,RegexpTokenizer
from nltk.corpus import stopwords
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC,LinearSVC
from nltk.classify import ClassifierI
from statistics import mode
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener


unpickle= open("lines.pickle", "rb")
lines = pickle.load(unpickle)
unpickle.close()




unpickle= open("5kcommonWords.pickle", "rb")
commonWords = pickle.load(unpickle)
unpickle.close()



#Function to return boolean value for words from the review which are amongst the most common words.
#Return True if the word in the review is among the top 4000 words.
def check_commonWords(words):
    words = word_tokenize(words)
    listOfCommonWords = {}
    for w in commonWords:
        listOfCommonWords[w] = (w in words)

    return listOfCommonWords


unpickle= open("featuresets.pickle", "rb")
featuresets = pickle.load(unpickle)
unpickle.close()

random.shuffle(featuresets)



#split the data into training and testing
training_set = featuresets[:10000]

# set that we'll test against.
testing_set = featuresets[10000:]


unpickle= open("naivebayes.pickle", "rb")
classifier = pickle.load(unpickle)
unpickle.close()



unpickle= open("logisticRegression.pickle", "rb")
LogisticRegression_classifier = pickle.load(unpickle)
unpickle.close()


unpickle= open("sgdClassifier.pickle", "rb")
SGDClassifier_classifier = pickle.load(unpickle)
unpickle.close()


unpickle= open("BernoulliClassifier.pickle", "rb")
BernoulliNB_classifier = pickle.load(unpickle)
unpickle.close()




unpickle= open("MNaivebayes.pickle", "rb")
MNB_classifier = pickle.load(unpickle)
unpickle.close()



'''
Aggregated Classifier takes different classifiers as input. To classify it takes vote from each classifier, and 
returns the class label with most number of votes(mode)
The get_confidence method returns the confidence on the vote(class label)

The get_confidence uses mode method from statistics which raisies statistics.StatisticsError
statistics.StatisticsError: no unique mode; found 2 equally common values 
Since number of classifiers used in uneven, this error is avoided.
'''
class AggregatedClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
    def classify(self, features):
            votes = []
            for classifier in self._classifiers:
                votes.append(classifier.classify(features))
            return mode(votes)
    def get_confidence(self, features):
        votes = []
        for classifier in self._classifiers:
            votes.append(classifier.classify(features))


        confidence = votes.count(mode(votes)) / len(votes)
        return confidence


agg_classifier = AggregatedClassifier(classifier,BernoulliNB_classifier,SGDClassifier_classifier,LogisticRegression_classifier,MNB_classifier)


def get_sentiment(text):
    features = check_commonWords(text)
    return agg_classifier.classify(features),agg_classifier.get_confidence(features)*100


