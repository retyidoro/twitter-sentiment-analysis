#pip install -U textblob nltk
from textblob.classifiers import NaiveBayesClassifier


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

#c1 = NaiveBayesClassifier(train)
#NaiveBayesClassifier.classify(text)

#print(c1.classify('The beer was good.'))
#print(c1.classify("I ain't feeling dandy today."))

#--------------------------------------tweetek train halmazba valogatasa--------------------------
train_tweets = []
with open("smalldata.txt") as f:
	for line in f:
		if line[0] == '1':
			sentiment = 'pos'
		else:
			sentiment = 'neg'
		train_tweets.append((line[2:-1], sentiment))

#print(train_tweets)

print("TRAINING")
print("---------------------------------------------")
c2 = NaiveBayesClassifier(train_tweets)
print("TRAINING DONE")
print("---------------------------------------------")
t = ["i like how you can change the colour of stuffs on bebo  amp how you can do the underline  italics amp bold thingys  haha",
"I like it when things work like they should I was expecting this upgrade to break something bigtime",
"i like katie wirth without brendon tooo",
"I like to eat out a lil too muchI guess I just like being served after a crazy day  workI need a frequent diner cardlol eeeek",
"I like you so much better when youre naked",
"i loove the songs of david archuleta  ",
"I love getting off work   and smelling the scent after it rains the best lt333",
"I love it tho lol  ",
"i love my fly with me friends  guyjbfanforlife and taylavision",
"i love singing along with my baby  our favorite song is the best   and sex was delicious lt3 hahaha",
"I love you  daddy"]
#t = ["Return and we lose",
s = ['pos', 'pos', 'neg', 'neg', 'pos', 'pos', 'pos', 'neg', 'pos', 'pos', 'pos']

test_tweets = []
sentiments = []
with open("dataset.txt") as f2:
	for line in f2:
		if line[0] == '1':
			sentiment = 'pos'
		else:
			sentiment = 'neg'
		test_tweets.append(line[2:-1])
		sentiments.append(sentiment)

 #"THE WEATHER WTF gt  Craving for a can of Monster ", "IceEmpress Ohhh I know ", "marshmelowsquid well belle sounds nice Cute ", "aha awesome  and its signed did it already come signed"]

print("The sentiment of the test tweets: ")
#result = c2.classify(testing)

k = 0;
db = 0
for k in range(100):
	if c2.classify(test_tweets[k]) == sentiments[k]:
		db = db + 1

print(db/100.0 * 100, "% ")
print("100 tweetbol ")
#print(s)
#print("---------")
#print(test_tweets)
#print(sentiments)

	

#tweetek kulon valogatasa
tweets_neg = [];
tweets_poz = [];
tweets= [];
with open('smalldata.txt', "r") as file:
	for line in file:
		if line[0] =='0':
			tweets_neg.append(line[2:]);
		else:
			tweets_poz.append(line[2:]);
		tweets.append(line[2:]);
	
neg = len(tweets_neg)	
poz = len(tweets_poz)
print('negativ tweetek szama a tanito halmazban:',neg)
print('pozitiv tweetek szama a tanito halmazban:',poz)
#N = len(tweets_neg) + len(tweets_poz)
N= len(tweets)
print('Osszesen ennyi tweet van a tanito halmazban',N)