'''
The following code uses the trained classifiers to predict the tweet sentiment
__author__="Soniya Rode"
'''
from tweepy.streaming import StreamListener
import json
import twitterClassifiers as classifier
from twitterDetails import *
from tweepy import Stream
from tweepy import OAuthHandler



class listener(StreamListener):

    def on_data(self, data):
        data = json.loads(data)
        tweet = data["text"]
        sentiment, confidence = classifier.get_sentiment(tweet)
        print(tweet," Sentiment :",sentiment,"Confidence: ", confidence)

        if confidence * 100 >= 80:
            output = open("twitterSentiments.txt", "a")
            output.write(sentiment)
            output.write('\n')
            output.close()

        return True

    def on_error(self, status):
        print("Error occured :" ,status)


auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
listOfWordsToTrack=["happy"]
twitterStream.filter(track=listOfWordsToTrack)
