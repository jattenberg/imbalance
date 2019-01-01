"""
   simple utilities for loading classifier datasets
"""
from sklearn.datasets import *
import numpy as np

def fetch_iris(positive=[2]):
    data = load_iris()
    return data.data, np.array([1 if x in positive else 0 for x in data.target])

def fetch_digits(positive=[8]):
    data = load_digits()
    return data.data, np.array([1 if x in positive else 0 for x in data.target])

def fetch_wine(positive=[2]):
    data = load_wine()
    return data.data, np.array([1 if x in positive else 0 for x in data.target])

def fetch_cancer():
    data = load_breast_cancer()
    return data.data, data.target

def fetch_faces(positive=[12]):
    data = fetch_olivetti_faces()
    return data.data, np.array([1 if x in positive else 0 for x in data.target])

def fetch_news(positive=[19]):
    data = fetch_20newsgroups_vectorized(remove=('headers', 'footers', 'quotes'))
    return data.data, np.array([1 if x in positive else 0 for x in data.target])

def sample_positive_examples(X, y, p=1.0):
    """
       samples from the positive examples with uniform probability
       todo: ensure at least 1 positive example
    """
    proba_mask = np.random.uniform(low=0.0, high=1.0, size=y.shape) <= p
    label_mask = y == 0
    mask = np.logical_or(proba_mask, label_mask)
    return X[mask], y[mask]


fetchers = {
    "iris"   : fetch_iris,
    "digits" : fetch_digits,
    "wine"   : fetch_wine,
    "cancer" : fetch_cancer,
    "faces"  : fetch_faces,
    "news"   : fetch_news
}
    
if __name__ == "__main__":
    for name, fetcher in fetchers.iteritems():
        print ("%s:" % name)
        X, y = fetcher()
        print (X.shape)
        print ("pos: %.3f, neg: %.3f" % (y.sum()/float(y.shape[0]), (y.shape[0] - y.sum())/float(y.shape[0])))        
