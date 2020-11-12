"""Prediction of Users based on tweet embeddings"""
 
import numpy as np
from sklearn.linear_model import LogisticRegression
from .models import User
from .twitter import vectorize_tweet

def predict_user(user0_name, user1_name, hypo_tweet_text):
    """
    Determines which user is more likely to say the hypo_tweet_text

    Example run: predict('elonmusk', 'nasa', 'Tesla cars are EV')
    Returns: either 0(for user0_name) or 1(for user1_name)
    """

    # Defining users
    user0 = User.query.filter(User.name ==user0_name).one()
    user1 = User.query.filter(User.name ==user1_name).one()

    # vectorizing their tweets
    user0_vects = np.array([tweet.vect for tweet in user0.tweets])
    user1_vects = np.array([tweet.vect for tweet in user1.tweets])

    # stacking the vectorized vects
    vects = np.vstack([user0_vects, user1_vects])

    # creating labels
    labels = np.concatenate(
        [np.zeros(len(user0.tweets)), 
        np.ones(len(user1.tweets))]
        )

    # Vectorizing hypothertical tweet
    hypo_tweet_vect = vectorize_tweet(hypo_tweet_text)

    # instantiate a predictor and fitting
    log_reg = LogisticRegression().fit(vects, labels)

    return log_reg.predict(hypo_tweet_vect.reshape(1, -1))

