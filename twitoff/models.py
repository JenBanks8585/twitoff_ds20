"""SQLAlchemy models and utility functions for Twitoff Application"""

from flask_sqlalchemy import SQLAlchemy



DB = SQLAlchemy()


# User table
class User(DB.Model):
    """Twitter User Table that will correspond to tweets - SQLAlchemy syntax"""
    id = DB.Column(DB.BigInteger, primary_key=True)  # id column (primary key)
    name = DB.Column(DB.String, nullable=False)  # name column
    newest_tweet_id = DB.Column(DB.BigInteger) # keeps track of recent tweet
    #screen_name = DB.Column(DB.String(128))
    #location = DB.Column(DB.String(128))
    #followers_count= DB.Column(DB.Integer)

    #def count_likes(self):
    #    count = 0
    #    for tweet in self.tweets:
    #        count += tweet.likes

    def __repr__(self):
        return "<User: {}>".format(self.name)


# Twitter table
class Tweet(DB.Model):
    """Tweet text data - associated with Users Table"""
    id = DB.Column(DB.BigInteger, primary_key=True)  # id column (primary key)
    text = DB.Column(DB.Unicode(300))
    vect = DB.Column(DB.PickleType, nullable= False)
    user_id = DB.Column(DB.BigInteger, DB.ForeignKey(
        "user.id"), nullable=False)
    #likes = DB.Column(DB.BigInteger)
    
    user = DB.relationship('User', backref=DB.backref('tweets', lazy=True))

    def __repr__(self):
        return "<Tweet: {}>".format(self.text)
