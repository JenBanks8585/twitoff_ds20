"""Retrieve tweets and users the create embeddings and populate database"""

from os import getenv
import tweepy
import spacy
from .models import DB, Tweet, User


TWITTER_API_KEY = getenv("TWITTER_API_KEY")
TWITTER_API_SECRET = getenv("TWITTER_API_SECRET")
TWITTER_AUTH = tweepy.OAuthHandler(TWITTER_API_KEY, TWITTER_API_SECRET)
TWITTER = tweepy.API(TWITTER_AUTH)

# nlp model
nlp = spacy.load('my_model')

def vectorize_tweet(tweet_texts):
    return nlp(tweet_texts).vector



def add_or_update_user(username):
    try:
        # getting a user, TWITTER is the API
        twitter_user = TWITTER.get_user(username)

        db_user = (User.query.get(twitter_user.id)) or User(
            id = twitter_user.id, name = username) 
        # adding the user in the database
        DB.session.add(db_user)

        # getting 200 tweets
        tweets = twitter_user.timeline(
            count = 200, 
            exclude_replies = True,
            include_rts = False, 
            tweet_mode = 'extended',
            since_id=db_user.newest_tweet_id
        )


        if tweets:
            db_user.newest_tweet_id = tweets[0].id


        # adding vectorized tweet in database
        for tweet in tweets:
            # stores numerical representations
            vectorized_tweet = vectorize_tweet(tweet.full_text)
            db_tweet = Tweet(id=tweet.id, text=tweet.full_text,
                             vect=vectorized_tweet)
            db_user.tweets.append(db_tweet)
            DB.session.add(db_tweet)


        # iteratively add user tweets to database
        #for tweet in tweets:
        #    db_tweet = Tweet(id = tweet.id, text=tweet.full_text)
        #    db_user.tweets.append(db_tweet)
        #    DB.session.add(db_tweet)

        #DB.session.commit()
    
    except Exception as e:
        #print error to user and raise throughout app
        print('Error processing {}: {}'.format(username, e))
        raise e

    # commits changes after try has completed
    else:
        DB.session.commit()

def update_all_users():
    """Update all Tweets for all Users in the User table."""
    for user in User.query.all():
        add_or_update_user(user.name)



