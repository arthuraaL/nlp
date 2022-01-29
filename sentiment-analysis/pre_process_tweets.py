"""The sample dataset from NLTK is separated into positive and negative tweets. 
It contains 5000 positive tweets and 5000 negative tweets exactly. The exact match between these classes is not a coincidence. 
The intention is to have a balanced dataset. That does not reflect the real distributions of positive and negative classes in live Twitter streams. 
It is just because balanced datasets simplify the design of most computational methods that are required for sentiment analysis.

This file can also be imported as a module and contains the following
functions:

    * get_dataset
    * pre_process_tweets

Remember to run the download module first.
"""


from typing import List, Tuple

import nltk # Python library for NLP
from nltk.corpus import twitter_samples # Twitter dataset from NLTK
from nltk.corpus import stopwords          # module for stop words that come with NLTK
from nltk.stem import PorterStemmer        # module for stemming
from nltk.tokenize import TweetTokenizer   # module for tokenizing strings

import matplotlib.pyplot as plt
import string
import re


def get_dataset(display: bool = True) -> Tuple(List[str], List[str]):
    """Function to get the dataset."""

    pos_tweets = twitter_samples.strings('positive_tweets.json')
    neg_tweets = twitter_samples.strings('negative_tweets.json')
    if display:
        print('Number of positive tweets: ', len(pos_tweets))
        print('Number of negative tweets: ', len(neg_tweets))
        print('\nThe type of all_positive_tweets is: ', type(pos_tweets))
        print('The type of a tweet entry is: ', type(pos_tweets[0]))

        fig = plt.figure(figsize=(7,7))
        labels=['Positives', 'Negatives']
        sizes = [len(pos_tweets), len(neg_tweets)]
        plt.bar(labels, sizes)
        plt.show()


    return pos_tweets, neg_tweets


def __clean(tweet: str) -> str:
    """Remove hyperlinks and signs"""

    cleaned_tweet = re.sub('http\S+', '', tweet)
    cleaned_tweet = re.sub('#', '', cleaned_tweet)
    return cleaned_tweet


def __tokenize(tweet: str) -> List[str]:
    """Tokenize the tweet. It is not necessary keep each word with capitalization."""

    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)
    return tweet_tokens


def __remove_stopwords_and_punctuation(tweet_tokens: List[str]) -> List[str]:
    """Remove stop words and punctuations. Let's retain some punctuation in the list that are important when dealing with tweets,
       like ':)' and '...', because they are used to express emotions."""

    stopwords_en = stopwords.words('english')
    puncts = string.punctuation
    new_tokens = []
    for word in tweet_tokens:
        if (word not in stopwords_en) and (word not in puncts):
            new_tokens.append(word)
    return new_tokens


def __stemming(tweet_tokens: List[str]) -> List[str]:
    """Stemming is the process of reducing inflected (or sometimes derived) words to their word stem, base or root form. For example:
       learning, learnt and learn have the word 'learn' as the stem. In this function we will use the PorterStemmer module with the 
       Porter stemming algorithm."""
       
    new_tokens = []
    stemmer = PorterStemmer()
    for word in tweet_tokens:
        stem_word = stemmer.stem(word, to_lowercase=True)
        new_tokens.append(stem_word)
    return new_tokens


def pre_process_tweets(tweet: str) -> List[str]:
    """Pre process tweets."""

    tweet = __clean(tweet)
    tweet_tokens = __tokenize(tweet)
    tweet_tokens = __remove_stopwords_and_punctuation(tweet_tokens)
    pre_processed_tweets = __stemming(tweet_tokens)

    return pre_processed_tweets


