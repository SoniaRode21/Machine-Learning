'''
The following code uses positive.txt and negative.txt data to train the classifiers for twitter analysis. The data files contain short sentences which are similar to tweets.
It does text preprocessing such as removing stop words and tokenization.
After preprocessing different classifiers like Naive Bayesian classifier,SVC_classifier,BernoulliNB_classifier,
SGDClassifier_classifier,LogisticRegression_classifier are modelled on the data and pickled.
Finally a aggregator classifer(vote) which outputs the most voted label and its confidence is implemented.


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



#Initalise the stop words set and tokenizer to remove punctuations.
stopwords = set(stopwords.words('english'))

tokenizer = RegexpTokenizer(r'\w+')
#read data from the files
lines=[]
posfile= open("positive.txt","r",encoding='latin-1').read()
for line in posfile.split('\n'):
    lines.append((line,"positive"))

print(lines[0])
negfile = open("negative.txt","r",encoding='latin-1').read()
for line in negfile.split('\n'):
    lines.append((line,"negative"))

pickleLines = open("lines.pickle","wb")
pickle.dump(lines,pickleLines )
pickleLines.close()

#Get all words in the movie reviews
words = []

pos_words = word_tokenize(posfile)
neg_words = word_tokenize(negfile)

for w in pos_words :
    if tokenizer.tokenize(w) and w not in stopwords:
        words.append(w.lower())
        #print(words)
for w in neg_words :
    if tokenizer.tokenize(w) and w not in stopwords:
        words.append(w.lower())


#Get frequencies of all words
words = nltk.FreqDist(words)


#Get top most 10 common words
words.most_common(10)
#Get the top 4000 common words
commonWords=list(words.keys())[:5000]
save_commonWords = open("5kcommonWords.pickle","wb")
pickle.dump(commonWords, save_commonWords)
save_commonWords.close()

#Function to return boolean value for words from the review which are amongst the most common words.
#Return True if the word in the review is among the top 4000 words.
def check_commonWords(words):
    words = word_tokenize(words)
    listOfCommonWords = {}
    for w in commonWords:
        listOfCommonWords[w] = (w in words)

    return listOfCommonWords

#Now
featuresets = [(check_commonWords(rev), category) for (rev, category) in lines]

Pickle_featuresets = open("featuresets.pickle","wb")
pickle.dump(featuresets, Pickle_featuresets)
Pickle_featuresets.close()


#split the data into training and testing
training_set = featuresets[:10000]

# set that we'll test against.
testing_set = featuresets[10000:]

#Use NLTK's Naive Bayes classifier
classifier = nltk.NaiveBayesClassifier.train(training_set)


print("Naive Bayesian Classifier accuracy:",(nltk.classify.accuracy(classifier, testing_set))*100)

classifier.show_most_informative_features(15)

#Store the classifier
Pickle_classifier = open("naivebayes.pickle","wb")
pickle.dump(classifier, Pickle_classifier)
Pickle_classifier.close()

#Logistic Regression Classifier
LogisticRegression_classifier = SklearnClassifier(LogisticRegression(solver='lbfgs'))
LogisticRegression_classifier.train(training_set)

#Store the logistic regression classifier
Pickle_classifier = open("logisticRegression.pickle","wb")
pickle.dump(LogisticRegression_classifier, Pickle_classifier)
Pickle_classifier.close()

#stochastic gradient descent (SGD) classifier
SGDClassifier_classifier = SklearnClassifier(SGDClassifier( max_iter=1000))
SGDClassifier_classifier.train(training_set)

#Store SGD classifier
Pickle_classifier = open("sgdClassifier.pickle","wb")
pickle.dump(SGDClassifier_classifier, Pickle_classifier)
Pickle_classifier.close()

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)

#Store SGD classifier
Pickle_classifier = open("BernoulliClassifier.pickle","wb")
pickle.dump(BernoulliNB_classifier, Pickle_classifier)
Pickle_classifier.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)

#Store the classifier
Pickle_classifier = open("MNaivebayes.pickle","wb")
pickle.dump(MNB_classifier, Pickle_classifier)
Pickle_classifier.close()

#Classifier Accuracies:
print("Classifier accuracies :")
print("Naive Bayes Classifier accuracy:",(nltk.classify.accuracy(classifier, testing_set))*100)
print("LogisticRegression_classifier accuracy:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)
print("SGDClassifier_classifier accuracy:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)



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

#print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)
for i in range(20):
    print("Predicted Class label:", agg_classifier.classify(testing_set[i][0]), "Confidence %:",agg_classifier.get_confidence(testing_set[i][0])*100)


def get_sentiment(text):
    features = check_commonWords(text)
    return agg_classifier.classify(features),agg_classifier.get_confidence(features)*100

print(get_sentiment("This movie was awesome! The acting was great, plot was wonderful, and there were pythons...so yea!"))



