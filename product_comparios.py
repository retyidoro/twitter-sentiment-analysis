from tkinter import *
from tkinter import messagebox
import json
from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream
import preprocessor as p
from textblob.classifiers import NaiveBayesClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
import re
import tkinter.messagebox
import sys
import os
import time
import matplotlib, numpy, sys
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from langdetect import detect
from sklearn.feature_extraction.text import CountVectorizer

from nltk.stem.porter import *
from nltk.corpus import stopwords

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report

ACCESS_TOKEN = '551761241-NLbQTMlZBssl68MeKo3N9P8RcsSYjqk0EHFNdosW'
ACCESS_SECRET = 'fkKDeBMhBKgKUF4sx2qoDjLIb0ciBA3sqxRLLWPg177Cl'
CONSUMER_KEY = 'PJVqgCUwP00ctxz5WY5zQjAmU'
CONSUMER_SECRET = 'ptQylfcYNykqTPIb3omVtcAJqWNz3S8qyL3lcqN8OXOY9rHCCX'
oauth = OAuth(ACCESS_TOKEN, ACCESS_SECRET, CONSUMER_KEY, CONSUMER_SECRET)


root = Tk()
root.title("Sentiment classification of tweets")
f = open('tweetek.txt', 'w')
lista = []
checklist = []
query1 = ""
query2 = ""
var1 = IntVar()
var2 = IntVar()

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
	
def plotting(res, p1, p2, name):
	#------------------new window plot --------------------------------------------
	window = tkinter.Toplevel(root)
	window.title(name)
	
	f = Figure(figsize=(4, 4), dpi=100);
	ax = f.add_subplot(111)

	data = [[p1, p2], [len(res)/2 - p1, len(res)/2 - p2]]
	ind = numpy.arange(2)
	
	rectsl = ax.bar(ind + 0.00, data[0], color='green', width=0.3, label="positive")
	rectsl = ax.bar(ind + 0.3, data[1], color='red', width=0.3, label="negative")
	
	ax.legend(bbox_to_anchor=(0, 1.02, 1, .102), loc=3, ncol=1, borderaxespad=0)
	ax.set_xticklabels([entry_1.get() , entry_3.get()])
	ax.set_xticks([0.15, 1.15])
	
	canvas = FigureCanvasTkAgg(f, window)
	canvas.show()
	canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)

	
def NaiveBayes():
	print('Bayes')
	train_tweets = [];
	sentiments = []
	test_tweets = lista
	#print(lista)
	
	with open("naive.txt") as f:
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
	res = []
	for k in range(len(test_tweets)):
		res.append(c2.classify(test_tweets[k]))


	resultLabels = [Variable() for i in res]
	p1 = 0
	p2 = 0
	szaml = 0
	print(res)
	for i in res:
		if szaml < len(res) / 2:
			if i == 'pos':
				p1 = p1 + 1;
		else:
			if i == 'pos':
				p2 = p2 + 1;
		szaml = szaml + 1;
	p = p1 + p2;
	
	
	j = 0
	gr = 0
	double = len(resultLabels) / 2
	for i in resultLabels:
		r = Label(frame, text=res[j])
		if j == double:
			gr = gr + 1;
		gr = gr + 1;
		r.grid(row= gr+1, column=2)
		j = j + 1
	
	
	percent=round(p/(j*1.0) *100.0, 2)
	labeltext = str(percent) + "% positive"
	if percent < 50.0:
		labeltext = str(100.0 - percent) + "% negative"
	
	
	r = Label(frame, text = labeltext)
	#r.grid(row=gr+2, column = 2)
	
	plotting(res, p1, p2, "Naive Bayes")
		
def Svm():
	print('Svm')
	classes = ['pos', 'neg']
	model = 1;
	if var1.get() == 1:
		print("bigram model")
		model = 2;
		
	

	# Read the data
	train_data = []
	train_labels = []
	test_data = []
	test_labels = []
	with open("svm_knn_train.txt") as f:
		for line in f:
			if line[0] == '1':
				train_labels.append('pos')
			else:
				train_labels.append('neg')
			train_data.append(line[2:-1].lower())

	test_data = lista
	# Create feature vectors
	
	vectorizer = CountVectorizer(ngram_range=(1, model), stop_words='english', token_pattern=r'\b\w+\b', min_df=5, max_df = 0.8)
	
	#vectorizer = TfidfVectorizer(min_df=5, max_df = 0.8, sublinear_tf=True, use_idf=True)
	train_vectors = vectorizer.fit_transform(train_data)
	test_vectors = vectorizer.transform(test_data)
	
	result = []
	
	classifier_liblinear = svm.LinearSVC()
	classifier_liblinear.fit(train_vectors, train_labels)
	result.append(classifier_liblinear.predict(test_vectors))
	
	res= [];
	res = result[0];
	resultLabels2 = [Variable() for i in res]
	
	p1 = 0
	p2 = 0
	szaml = 0
	print(res)
	for i in res:
		if szaml < len(res) / 2:
			if i == 'pos':
				p1 = p1 + 1;
		else:
			if i == 'pos':
				p2 = p2 + 1;
		szaml = szaml + 1;
	p = p1 + p2;
	
	double = len(resultLabels2) / 2;
	print(double)
	j = 0
	gr = 0
	for i in resultLabels2:
		r = Label(frame, text=res[j])
		if j == double:
			gr = gr + 1;
		gr = gr + 1;
		r.grid(row= gr+1, column=3)
		j = j + 1
		
	
	percent=round(p/(j*1.0) *100.0, 2)
	labeltext = str(percent) + "% positive"
	if percent < 50.0:
		labeltext = str(100.0 - percent) + "% negative"
	
	
	r = Label(frame, text = labeltext)
	#r.grid(row=gr+2, column = 3)
		
	#prediction_liblinear = classifier_liblinear.predict(test_vectors)	
	#print(classification_report;(test_labels, prediction_liblinear))	
	cimke = "SVM"
	if model == 2:
		cimke = "SVM bi-gram"
	plotting(res, p1, p2, cimke)

	
def Knn():
	classes = ['pos', 'neg']
	
	model = 1;
	if var2.get() == 1:
		print("bigram model")
		model = 2;

	# Read the data
	train_data = []
	train_labels = []
	test_data = []
	test_labels = []
	with open("svm_knn_train.txt") as f:
		for line in f:
			if line[0] == '1':
				train_labels.append('pos')
			else:
				train_labels.append('neg')
			train_data.append(line[2:-1].lower())

	test_data = lista
	# Create feature vectors
	
	#vectorizer = TfidfVectorizer(min_df=5, max_df = 0.8, sublinear_tf=True, use_idf=True)
	vectorizer = CountVectorizer(ngram_range=(1, model), stop_words='english', token_pattern=r'\b\w+\b', min_df=5, max_df = 0.8)
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
	
	p1 = 0
	p2 = 0
	szaml = 0
	print(res)
	for i in res:
		if szaml < len(res) / 2:
			if i == 'pos':
				p1 = p1 + 1;
		else:
			if i == 'pos':
				p2 = p2 + 1;
		szaml = szaml + 1;
	p = p1 + p2;
	
	j = 0
	gr = 0
	double=len(resultLabels3) / 2
	for i in resultLabels3:
		r = Label(frame, text=res[j])
		if j == double:
			gr = gr + 1;
		gr = gr + 1;
		r.grid(row= gr+1, column=4)
		j = j + 1
	
	
	percent=round(p/(j*1.0) *100.0, 2)
	percent2 = round(p/(j*1.0) *100)
	labeltext = str(percent) + "% positive"
	if percent < 50.0:
		labeltext = str(100.0 - percent) + "% negative"
	
	
	r = Label(frame, text = labeltext)
	#r.grid(row=gr+2, column = 4)
	
	cimke = "Knn"
	if model == 2:
		cimke = "Knn bi-gram"
	plotting(res, p1, p2, cimke)
	
	
def getTweets(query, tweet_count, lista, szoveg, twitter_stream):
	iterator = twitter_stream.statuses.filter(track=query, languages='en')
	
	for tweet in iterator:
	
		
		print("-----------------------------------")
		data = json.loads(json.dumps(tweet));
		text = data["text"]
		#text = ' RT @valaki #alma kotre www.alama.com https://python.org'
		re00 = re.sub(r'\\', '', text)
		re001 = re.sub(r'http\S+', '', re00)
		
		
		
		
		re002 = re.sub(r'www.\S+', '', re001)
		#p.set_options(p.OPT.URL, p.OPT.EMOJI)
		#cleantweet= p.clean(re002)
		#
		re1 = re.sub(r'@\S+', '', re002)
		re2 = re.sub(r'RT', '', re1)
		re2 = re.sub(r'\:\)', 'happy', re2)
		re2 = re.sub(r'\:D', 'happy', re2)
		re2 = re.sub(r'\:d', 'happy', re2)
		re2 = re.sub(r'\:P', 'happy', re2)
		re2 = re.sub(r'\:p', 'happy', re2)
		re2 = re.sub(r'\;D', 'happy', re2)
		re2 = re.sub(r'\;d', 'happy', re2)
		re2 = re.sub(r'\;p', 'happy', re2)
		re2 = re.sub(r'\:P', 'happy', re2)
		re2 = re.sub(r'\:\-\)', 'happy', re2)
		re2 = re.sub(r'\;\-\)', 'happy', re2)
		re2 = re.sub(r'\:\=\)', 'happy', re2)
		re2 = re.sub(r'\;\=\)', 'happy', re2)
		re2 = re.sub(r'\:\<\)', 'happy', re2)
		re2 = re.sub(r'\:\>\)', 'happy', re2)
		re2 = re.sub(r'\;\>\)', 'happy', re2)
		re2 = re.sub(r'\;\-3', 'happy', re2)
		re2 = re.sub(r'\:\-3', 'happy', re2)
		re2 = re.sub(r'\;\-x', 'happy', re2)
		re2 = re.sub(r'\:\-x', 'happy', re2)
		re2 = re.sub(r'\:\-X', 'happy', re2)
		re2 = re.sub(r'\;\-X', 'happy', re2)
		re2 = re.sub(r'\^\_\^', 'happy', re2)
		re2 = re.sub(r'\^-^', 'happy', re2)
		re2 = re.sub(r'\:\-\]', 'happy', re2)
		re2 = re.sub(r'\:\-\.', 'happy', re2)
		re2 = re.sub(r'\;\'\)', 'happy', re2)
		
		re2 = re.sub(r'\:\(', 'sad', re2)
		re2 = re.sub(r'\;\(', 'sad', re2)
		re2 = re.sub(r'\:\'\(', 'sad', re2)
		re2 = re.sub(r'\)\:', 'sad', re2)
		re2 = re.sub(r'\)\;', 'sad', re2)
		re2 = re.sub(r'\)\'\\;', 'sad', re2)
		re2 = re.sub(r'\)\'\\:', 'sad', re2)
		re2 = re.sub(r'\:\-\(', 'sad', re2)
		re2 = re.sub(r'\;\-\(', 'sad', re2)
		re2 = re.sub(r'\[\:', 'sad', re2)
		re2 = re.sub(r'\;\]', 'sad', re2)
		re2 = re.sub(r'T_T', 'sad', re2)
		re2 = re.sub(r'\:\-\/', 'sad', re2)
		re2 = re.sub(r'\;\-\(', 'sad', re2)
		re3 = re.sub(r'(.)\1+', r'\1\1', re2) 
		
		
		
		for word in re3.split():
			if word.startswith("#"):
				print(word)
				idx = re3.find(word)
				re3 = re001[:idx] + " " + word + " " + re3[idx:]

		
		re4 = re.sub(r'[^a-zA-Z, ]','',re3)
		
		
		
		
		
		print(text)


		if not re4:
			print('ures')
		else:
			try:
				if detect(re4) == 'en':
					
					tweet_count -= 1
					szoveg = szoveg + str(re4) + '\n'
					#filtered = [t for t in re4.split() if t not in stop_words]
					#stemmed_szavak = [stemmer.stem(szo) for szo in re4.split()]
					#print(" ".join(stemmed_szavak))
					#lista.append(" ".join(stemmed_szavak))
					lista.append(re4)
			except:
				print("nem detektalhato nyelv")
		
		
			
		
		
		if tweet_count <= 0:
			break 
	

def preprocessing():
	global lista;
	
	query1 = entry_1.get()
	query2 = entry_3.get()
	if len(entry_1.get()) == 0 or len(entry_3.get()) == 0:
		messagebox.showinfo("Ooops", "Please enter a query!")
		return
		

	count = entry_2.get()
	if len(entry_2.get()) == 0:
		messagebox.showinfo("Ooops", "Please enter a number!")
		return
	
	try:
		tweet_count = int(count)
	except:
		messagebox.showinfo("Ooops", "Please enter a number!")
		return
		
	double = tweet_count
	tweet_count2 = double
	print('Saving tweets')
	
	szoveg=""
	labellist=[Variable() for i in range(2 * double + 1)]
	
	# Initiate the connection to Twitter Streaming API
	twitter_stream = TwitterStream(auth=oauth)
	
	getTweets(query1, tweet_count, lista, szoveg, twitter_stream)
	
	
	getTweets(query2, tweet_count, lista, szoveg, twitter_stream)
	
	#----------------tweets
	#print(szoveg)
	
	frame.grid(columnspan=4)
	label1 = Label(frame, text= query1, font='Helvetica 12 bold')
	
	label1.grid(row= 1, column=0)
	
	
	j=0
	for i in range(double):
		l = Label(frame, text=lista[j])
		l.grid(row=j+2, column=0)
		j = j + 1

	labell2 = Label(frame, text = query2, font='Helvetica 12 bold')
	labell2.grid(row= j+2, column=0)
	
	for i in range(double):
		#print(query2)
		l = Label(frame, text=lista[j])
		l.grid(row=j+3, column=0)
		j = j + 1
	
	#------------checkbuttons
	label2 = Label(frame, text="Check if positive ")
	label2.grid(row= 1, column=1)
	
	label3 = Label(frame, text="Bi-gram model")
	label3.grid(row= 0, column=1)
	
	checklist = [IntVar() for i in range(2* double)]
	for i in checklist:
		i.set(0)
	
	
	
	#print(double)
	j = 1;
	for i in checklist:
		if j == double + 1:
			j = j + 1;
		c = Checkbutton(frame, variable=i, command=checked)
		c.grid(row=j+1, column=1)
		
		j = j + 1

	
	buttonProcess.config(state=DISABLED)	
	#-----------algorithms
	buttonBayes = Button(frame, text='Naive Bayes', command=NaiveBayes)
	buttonBayes.grid(row=1, column=2)
	
	
	buttonSvm = Button(frame, text='Svm', command=Svm)
	buttonSvm.grid(row = 1, column=3)
	
	Checkbutton1 = Checkbutton(frame, variable=var1)
	Checkbutton1.grid(row=0, column=3)
	#
	
	buttonKnn = Button(frame, text='Knn', command=Knn)
	buttonKnn.grid(row = 1, column=4)
	Checkbutton2 = Checkbutton(frame, variable=var2)
	Checkbutton2.grid(row=0, column=4)
	
	#
	
	
	



label_1 = Label(root, text='query:')
label_2 = Label(root, text='number of tweets:')
label_3 = Label(root, text='VS')
entry_1 = Entry(root)
entry_2 = Entry(root)
entry_3 = Entry(root)
buttonProcess = Button(root, text='Get tweets', command=preprocessing)


label_1.grid(row=0, sticky=E)
label_2.grid(row=1, sticky=E)
entry_1.grid(row=0, column=1)
entry_2.grid(row=1, column=1)
label_3.grid(row=0, column=2)
entry_3.grid(row=0, column=3)
buttonProcess.grid(columnspan=5)


frame = Frame(root)
root.mainloop()
