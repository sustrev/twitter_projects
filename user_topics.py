import pandas as pd
import tweepy
import twitter_secrets
import logging
import sys
from gensim.parsing.preprocessing import preprocess_string, STOPWORDS
from gensim.models.phrases import Phrases, Phraser
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import re
from nltk.corpus import stopwords
import nltk
#nltk.download('stopwords')

consumer_key, consumer_secret, access_token, access_token_secret, bearer_token = twitter_secrets.twitter_secrets()

logging.basicConfig(filename='loggerfile.log', encoding='utf-8', level=logging.WARNING)

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

def retrieve_tweets(user_handle, n):
    """
    Retrieves the n most recent tweets by an user_handle
    Returns a list of tweepy Status objects
    """
    user_feeds = []
    for status in tweepy.Cursor(api.user_timeline, id=user_handle).items(n):
        user_feeds.append(status)
    
    return user_feeds
    
def create_tweets_dataframe(tweets):
    """
    Creates a dataframe from a given sequence of tweets, each represented by one tweepy Status object.
        'id_str', string
        'retweet_count', int
        'created_at', datetime
        'text', string
        'lang', string
    """
    df_array = []
    for tweet in tweets:
        df_array.append(tweet._json)
    t_df = pd.DataFrame(df_array, columns =['retweet_count','created_at','text',
                                            'lang', 'id_str'])
    t_df['created_at'] = pd.to_datetime(t_df['created_at'])

    return t_df

def clean_and_tokenize(tweet_df):
    """
    Takes a tweet dataframe
    Cleans and tokenizes the text column
    Returns dataframe with tokenized column
    """
    tweet_df['text'] = tweet_df['text'].astype(str)
    tweet_df['tokens'] = tweet_df['text'].str.lower()
    tweet_df['tokens'] = tweet_df.tokens.apply(preprocess_string)
    
    # remove custom stopwords
    CUSTOM_STOP_WORDS = ['www','tinyurl','com', 'https', 'http', '&amp', 'rt', 'bit', 'ly', 'bitly']
    tweet_df['tokens'] = tweet_df.tokens.apply(lambda x: [w for w in x if w not in CUSTOM_STOP_WORDS])
    
    # remove common stopwords
    common_stopwords = stopwords.words('english')
    tweet_df['tokens'] = tweet_df.tokens.apply(lambda x: [w for w in x if w not in common_stopwords])
    
    # remove gensim stopwords
    tweet_df['tokens'] = tweet_df.tokens.apply(lambda x: [w for w in x if w not in STOPWORDS])
    
    # add bigrams, if applicable
    phrase_model = Phrases(tweet_df.tokens, threshold=20)
    frozen_model = phrase_model.freeze()
    
    tweet_df['token_bigrams'] = frozen_model[tweet_df.tokens]
    
    tweet_df['tokens'] = tweet_df.apply(lambda x: list(set(x['tokens']).union(set(x['token_bigrams']))), axis=1)
    tweet_df.drop(columns='token_bigrams', inplace=True)
    
    return tweet_df

def find_topics(df, num_topics):
    """
    Uses gensim's LDA model to identify top # of topics by token
    """
    tokens = df['tokens']
    
    # use gensim's Dictionary to filter words that appear less than ten times in the corpus
    # or represent more than 60% of the corpus
    dictionary = Dictionary(tokens)
    dictionary.filter_extremes(no_below=10, no_above=.6)
    
    # use the dictionary to create a bag of word representation of each document
    corpus = [dictionary.doc2bow(text) for text in tokens]
    
    # create gensim's LDA model 
    lda_model = LdaModel(corpus, id2word=dictionary, num_topics=num_topics, chunksize=2000, passes=20, iterations=400, eval_every=None, random_state=42,alpha='auto',eta='auto')
    top_topics = lda_model.top_topics(corpus=corpus)
    
    topic_tokens = []
    for tt in top_topics:
        _, topic = tt[0][0]
        if topic not in topic_tokens:
            topic_tokens.append(topic)
        
    return topic_tokens

def topic_top_tweet(df, topic):
    filtered_df = df[df['tokens'].apply(lambda x: topic in x)]
    sorted_df = filtered_df.sort_values('retweet_count', ascending=False)
    return sorted_df.head(1)

def pretty_print(df, top_topics):
    full_output = ""
    for topic in top_topics:
        focus_tweet = topic_top_tweet(df, topic)
        text = focus_tweet['text'].values[0]
        tweet_id = focus_tweet['id_str'].values[0]
        url = "https://twitter.com/user/status/{}".format(tweet_id)
        topic_output = """        Topic Token: {}
Top Retweeted Tweet: {}
               Link: {}

""".format(topic, text, url)
        full_output = full_output + topic_output
        
    return full_output

def talks_about(user):
    tweets = retrieve_tweets(user, 200)
    tweets_df = create_tweets_dataframe(tweets)
    token_tweets = clean_and_tokenize(tweets_df)
    top_topics = find_topics(token_tweets, 10)
    output = pretty_print(token_tweets, top_topics)
    
    return output

def main():
    try:
        user_id = sys.argv[1]
        print("What has {} been talking about lately?".format(user_id))
        print(talks_about(user_id))
    except:
        print("Please re-run and remember to specify a valid Twitter user!")
    
if __name__ =="__main__":
    main()