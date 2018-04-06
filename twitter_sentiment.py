from Tkinter import *
import json
from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream
import preprocessor as p
from textblob.classifiers import NaiveBayesClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
import re
import tkMessageBox
import sys
import os
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report

ACCESS_TOKEN = 'your acces token'
ACCESS_SECRET = 'your secret access'
CONSUMER_KEY = 'your consumer key'
CONSUMER_SECRET = 'your consumer secret'

oauth = OAuth(ACCESS_TOKEN, ACCESS_SECRET, CONSUMER_KEY, CONSUMER_SECRET)


root = Tk()
root.title("Sentiment classification of tweets")
f = open('tweetek.txt', 'w')
lista = []
checklist = []

def checked():
	#print("do nothing")
	l = []

	for i in checklist:
		if i.get() == 1:
			l.append('pos')
		else:
			print(i.get())
			l.append('neg')
	
	#print(l)
	
def NaiveBayes():
	print('Bayes')
	train_tweets = [];
	sentiments = []
	test_tweets = lista
	#print(lista)
	
	with open("smalldata1.txt") as f:
		for line in f:
			if line[0] == '1':
				sentiment = 'pos'
			else:
				sentiment = 'neg'
			train_tweets.append((line[2:-1].lower(), sentiment))
			sentiments.append(sentiment)

	#print(train_tweets)

	print("TRAINING")
	print("---------------------------------------------")
	c2 = NaiveBayesClassifier(train_tweets)
	print("TRAINING DONE")
	print("---------------------------------------------")
	
	
	k = 0;
	result = []
	for k in range(len(test_tweets)):
		result.append(c2.classify(test_tweets[k]))


	resultLabels = [Variable() for i in result]
	p = 0
	for i in result:
		if i == 'pos':
			p = p + 1;
	
	
	j = 0
	for i in resultLabels:
		r = Label(frame, text=result[j])
		r.grid(row= j+1, column=2)
		j = j + 1
	
	
	percent=p/(j*1.0) *100.0
	labeltext = str(percent) + "% positive"
	if percent < 50.0:
		labeltext = str(100.0 - percent) + "% negative"
	
	
	r = Label(frame, text = labeltext)
	r.grid(row=j+1, column = 2)
		
def Svm():
	print('Svm')
	classes = ['pos', 'neg']

	# Read the data
	train_data = []
	train_labels = []
	test_data = []
	test_labels = []
	with open("smalldata.txt") as f:
		for line in f:
			if line[0] == '1':
				train_labels.append('pos')
			else:
				train_labels.append('neg')
			train_data.append(line[2:-1].lower())

	test_data = lista
	# Create feature vectors
	
	vectorizer = TfidfVectorizer(min_df=5,
								max_df = 0.8,
								sublinear_tf=True,
								use_idf=True)
	train_vectors = vectorizer.fit_transform(train_data)
	test_vectors = vectorizer.transform(test_data)
	
	result = []
	
	classifier_liblinear = svm.LinearSVC()
	classifier_liblinear.fit(train_vectors, train_labels)
	result.append(classifier_liblinear.predict(test_vectors))
	
	res= [];
	res = result[0];
	resultLabels2 = [Variable() for i in res]
	
	p = 0
	for i in res:
		if i == 'pos':
			p = p + 1;
	
	j = 0
	for i in resultLabels2:
		r = Label(frame, text=res[j])
		r.grid(row= j+1, column=3)
		j = j + 1
	
	percent=p/(j*1.0) *100.0
	labeltext = str(percent) + "% positive"
	if percent < 50.0:
		labeltext = str(100.0 - percent) + "% negative"
	
	
	r = Label(frame, text = labeltext)
	r.grid(row=j+1, column = 3)
		
	#prediction_liblinear = classifier_liblinear.predict(test_vectors)	
	#print(classification_report(test_labels, prediction_liblinear))	
	
def Knn():
	classes = ['pos', 'neg']

	# Read the data
	train_data = []
	train_labels = []
	test_data = []
	test_labels = []
	with open("smalldata.txt") as f:
		for line in f:
			if line[0] == '1':
				train_labels.append('pos')
			else:
				train_labels.append('neg')
			train_data.append(line[2:-1].lower())

	test_data = lista
	# Create feature vectors
	
	vectorizer = TfidfVectorizer(min_df=5,
								max_df = 0.8,
								sublinear_tf=True,
								use_idf=True)
	train_vectors = vectorizer.fit_transform(train_data)
	test_vectors = vectorizer.transform(test_data)
	
	#print(test_vectors)
	
	neigh = KNeighborsClassifier(n_neighbors=3)
	neigh.fit(train_vectors,train_labels)
	KNeighborsClassifier
	
	m = neigh.predict_proba(test_vectors)
	#print(m) 
	res = [];
	for i in m:
		#print(i)
		if i[0] > i[1]:
			res.append('neg')
		else:
			res.append('pos')
		

	resultLabels3 = [Variable() for i in res]
	
	p = 0
	for i in res:
		if i == 'pos':
			p = p + 1;
	
	j = 0
	for i in resultLabels3:
		r = Label(frame, text=res[j])
		r.grid(row= j+1, column=4)
		j = j + 1
	
	
	percent=p/(j*1.0) *100.0
	labeltext = str(percent) + "% positive"
	if percent < 50.0:
		labeltext = str(100.0 - percent) + "% negative"
	
	
	r = Label(frame, text = labeltext)
	r.grid(row=j+1, column = 4)

def preprocessing():
	global lista;
	
	query = entry_1.get()
	if len(entry_1.get()) == 0:
		tkMessageBox.showinfo("Ooops", "Please enter a query!")
		return
		

	count = entry_2.get()
	if len(entry_2.get()) == 0:
		tkMessageBox.showinfo("Ooops", "Please enter a number!")
		return
	
	try:
		tweet_count = int(count)
	except:
		tkMessageBox.showinfo("Ooops", "Please enter a number!")
		return
		
	double = tweet_count
	print('Saving tweets')
	
	szoveg=""
	labellist=[Variable() for i in range(double)]
	
	# Initiate the connection to Twitter Streaming API
	twitter_stream = TwitterStream(auth=oauth)
	iterator = twitter_stream.statuses.filter(track=query, languages='en')
	for tweet in iterator:
	
		print("-----------------------------------")
		data = json.loads(json.dumps(tweet));
		text = data["text"].encode('utf-8')
		re00 = re.sub(r'\\', '', text)
		re001 = re.sub(r'http\S+', '', re00)
		re002 = re.sub(r'www.\S+', '', re001)
		p.set_options(p.OPT.URL, p.OPT.EMOJI)
		cleantweet= p.clean(re002)
		
		re1 = re.sub(r'@\S+', '', cleantweet)
		re2 = re.sub(r'RT', '', re1) #retweeted
		re3 = re.sub(r'[^a-zA-Z,.?!;: ]','',re2)
		re4 = re.sub(r'[,.?!:;]',' ',re3)
		re5 = re.sub(r'(.)\1+', r'\1\1', re4).lower()   
	
		if re4 or not re5.isspace():
			lista.append(re5)
			tweet_count -= 1
		
		szoveg = szoveg + str(re5) + '\n'
		if tweet_count <= 0:
			break 
			
	#----------------tweets
	#print(szoveg)
	
	frame.grid(columnspan=4)
	label1 = Label(frame, text= "tweets")
	label1.grid(row= 0, column=0)
	
	j=0
	for i in labellist:
		l = Label(frame, text=lista[j])
		l.grid(row=j+1, column=0)
		j = j + 1

	
	#------------checkbuttons
	label2 = Label(frame, text="Check if positive ")
	label2.grid(row= 0, column=1)
	
	checklist = [IntVar() for i in range(double)]
	for i in checklist:
		i.set(0)
	
	#print(double)
	j = 1;
	for i in checklist:
		c = Checkbutton(frame, variable=i, command=checked)
		c.grid(row=j, column=1)
		j = j + 1

	buttonProcess.config(state=DISABLED)	
	#-----------algorithms
	buttonBayes = Button(frame, text='Naive Bayes', command=NaiveBayes)
	buttonBayes.grid(row=0, column=2)
	
	
	buttonSvm = Button(frame, text='Svm', command=Svm)
	buttonSvm.grid(row = 0, column=3)
	
	buttonKnn = Button(frame, text='Knn', command=Knn)
	buttonKnn.grid(row = 0, column=4)



label_1 = Label(root, text='query:')
label_2 = Label(root, text='number of tweets:')
entry_1 = Entry(root)
entry_2 = Entry(root)
buttonProcess = Button(root, text='Get tweets', command=preprocessing)


label_1.grid(row=0, sticky=E)
label_2.grid(row=1, sticky=E)
entry_1.grid(row=0, column=1)
entry_2.grid(row=1, column=1)
buttonProcess.grid(columnspan=5)


frame = Frame(root)

root.mainloop()