import os
import nltk
from nltk import *
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

def parse_amazon_data(file_path):
	'''
	This function parses amazon data. It retrieves the reviews and good/bad label.

	:param file_path: path to traing/testing files
			-./data/test.ft.txt
			-./data/train.ft.txt
	:return: a tuple consisting of
			1. a list of reviews (a list of strings)
			2. a list of good/bad labels (stored as ints. 0 = bad, 1 = good)
	'''
	stopwords_set = set(stopwords.words('english'))
	lemmatizer = WordNetLemmatizer().lemmatize

	review_str_list = []
	labels_list = []

	with open(file_path, 'r') as f:
		words = []
		for line in f:
			label_str, review_str = line.split(' ', 1)
			label = int(label_str[-1]) - 1		# original data is labeled with 1 and 2. change to labels of 0 and 1
			# words = [lemmatizer(w) for w in review_str.lower().split() if w not in stopwords_set]	# get the words out
			review_str_list.append(review_str_list.lower())
		
			# Not too sure what the best DS is here but gonna store everything in list for now
			# review_str_list.append(words)
			labels_list.append(label)

	return (review_str_list, labels_list)

def parse_amazon_data_count_matrix(file_path, count_vect, training):
	'''
	This function parses amazon data. It retrieves the reviews and good/bad label.

	:param file_path: file_path: path to traing/testing files
			-./data/test.ft.txt
			-./data/train.ft.txt
	:param count_vect: vectorizer
	:param training: 1 if training data
					 0 if testing data
	:return:
		a sparse matrix w/ each row being a document and coluumn being a word
	'''

	review_str_list = []
	labels_list = []

	with open(file_path, 'r') as f:
		words = []
		for line in f:
			label_str, review_str = line.split(' ', 1)
			label = int(label_str[-1]) - 1  # original data is labeled with 1 and 2. change to labels of 0 and 1
			words = review_str

			# Not too sure what the best DS is here but gonna store everything in list for now
			review_str_list.append(words)
			labels_list.append(label)

	if training:
		X_counts = count_vect.fit_transform(review_str_list)
	else:
		X_counts = count_vect.transform(review_str_list)

	return (X_counts, labels_list)

def parse_imdb_data(file_path):
	'''

	:param file_path: file_path: path to training/testing files
			-./data/aclImdb/train
			-./data/aclImdb/test
	:return: a tuple consisting of
			1. a list of reviews (a list of strings)
			2. a list of good/bad labels (stored as ints. 0 = bad, 1 = good)
	'''
	stopwords_set = set(stopwords.words('english'))
	lemmatizer = WordNetLemmatizer().lemmatize

	review_str_list = []
	labels_list = []

	for folder_name in os.listdir(file_path):
		if folder_name == "pos":
			label = 1
		elif folder_name == "neg":
			label = 0
		else:
			# We are not interested in these data!!
			continue

		for file_name in os.listdir(file_path + "/" + folder_name):
			with open(file_path + "/" + folder_name + "/" + file_name, 'r', errors='ignore') as f:
				words = []
				for line in f:
					words += [lemmatizer(w) for w in line.lower().split() if w not in stopwords_set]
			review_str_list.append(words)
			labels_list.append(label)

	return (review_str_list, labels_list)


def parse_imdb_data_count_matrix(file_path, count_vect, training):
	'''

	:param file_path: path to training/testing files
			-./data/aclImdb/train
			-./data/aclImdb/test
	:param count_vect: vectorizer
	:param training: 1 if training data
					 0 if testing data
	:return:
		a sparse matrix w/ each row being a document and coluumn being a word
	'''

	review_str_list = []
	labels_list = []

	for folder_name in os.listdir(file_path):
		if folder_name == "pos":
			label = 1
		elif folder_name == "neg":
			label = 0
		else:
			# We are not interested in these data!!
			continue

		for file_name in os.listdir(file_path + "/" + folder_name):
			with open(file_path + "/" + folder_name + "/" + file_name, 'r', errors='ignore') as f:
				words = ""
				for line in f:
					words += " " + line
			review_str_list.append(words)
			labels_list.append(label)

	if training:
		X_counts = count_vect.fit_transform(review_str_list)
	else:
		X_counts = count_vect.transform(review_str_list)

	return (X_counts, labels_list)

def naive_bayes(x, y):
	nb = MultinomialNB()
	nb = nb.fit(x, y)
	return nb

if __name__ == "__main__":
	training_amazon = "./data/train.ft.txt"
	testing_amazon = "./data/test.ft.txt"
	training_imdb = "./data/aclImdb/train"
	testing_imdb = "./data/aclImdb/test"

	# train on amazon -> test on imdb
	print("train on amazon, test on imdb")

	# uncomment to run naive bayes
	count_vect = CountVectorizer(stop_words='english')

	amazon_training_x, amazon_training_y = parse_amazon_data_count_matrix(training_amazon, count_vect, 1)
	print("loaded amazon training data")
	amazon_testing_x, amazon_testing_y = parse_amazon_data_count_matrix(testing_amazon, count_vect, 0)
	print("loaded amazon testing data")
	imdb_testing_x, imdb_testing_y = parse_imdb_data_count_matrix(testing_imdb, count_vect, 0)
	print("loaded imdb testing data")


	nb1 = naive_bayes(amazon_training_x, amazon_training_y)
	print("classifier trained")
	amazon_pred_label1 = nb1.predict(amazon_testing_x)
	print("amazon predictions made")
	amazon_score1 = metrics.accuracy_score(amazon_pred_label1, amazon_testing_y)
	print("amazon score = " + str(amazon_score1))
	imdb_pred_label1 = nb1.predict(imdb_testing_x)
	print("imdb predictions made")
	imdb_score1 = metrics.accuracy_score(imdb_pred_label1, imdb_testing_y)
	print("amazon score = " + str(imdb_score1))


	#####################################################################################################
	# train on imdb -> test on amazon
	print("train on imdb, test on amazon")

	# uncomment to run naive bayes
	count_vect = CountVectorizer(stop_words='english')

	imdb_training_x, imdb_training_y = parse_imdb_data_count_matrix(training_imdb, count_vect, 1)
	print("loaded imdb training data")
	imdb_testing_x, imdb_testing_y = parse_imdb_data_count_matrix(testing_imdb, count_vect, 0)
	print("loaded imdb testing data")
	amazon_testing_x, amazon_testing_y = parse_amazon_data_count_matrix(testing_amazon, count_vect, 0)
	print("loaded amazon testing data")


	nb2 = naive_bayes(imdb_training_x, imdb_training_y)
	print("classifier trained")
	imdb_pred_label2 = nb2.predict(imdb_testing_x)
	print("imdb predictions made")
	imdb_score2 = metrics.accuracy_score(imdb_pred_label2, imdb_testing_y)
	print("imdb score = " + str(imdb_score2))
	amazon_pred_label2 = nb2.predict(amazon_testing_x)
	print("amazon predictions made")
	amazon_score2 = metrics.accuracy_score(amazon_pred_label2, amazon_testing_y)
	print("amazon score = " + str(amazon_score2))
