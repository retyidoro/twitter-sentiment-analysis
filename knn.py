from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

train = [
    ('I love this sandwich.', 'pos'),
    ('This is an amazing place!', 'pos'),
    ('I feel very good about these beers.', 'pos'),
    ('This is my best work.', 'pos'),
    ("What an awesome view", 'pos'),
    ('I do not like this restaurant', 'neg'),
    ('I am tired of this stuff.', 'neg'),
    ("I can't deal with this", 'neg'),
    ('He is my sworn enemy!', 'neg'),
    ('My boss is horrible.', 'neg')
]
test = [
    ('The beer was good.', 'pos'),
    ('I do not enjoy my job', 'neg'),
    ("I ain't feeling dandy today.", 'neg'),
    ("I feel amazing!", 'pos'),
    ('Gary is a friend of mine.', 'pos'),
    ("I can't believe I'm doing this.", 'neg')
]

#values = ['pos', 'neg']

samples = [[1, 2, 3], [4, 5, 2], [6, 2, 3], [1, 2, 1], [3, 2, 5], [3, 4, 2]]
values = [0, 1, 0, 0, 1, 1]

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(samples,values)
KNeighborsClassifier
#NearestNeighbors(algorithm='auto', leaf_size=30)

print(neigh.predict_proba([[3, 1, 5]])) 