import os
import nltk
from nltk import *
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

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
			words = [lemmatizer(w) for w in review_str.lower().split() if w not in stopwords_set]	# get the words out
		
			# Not too sure what the best DS is here but gonna store everything in list for now
			review_str_list.append(words)
			labels_list.append(label)

	return (review_str_list, labels_list)

def parse_amazon_data_sklearn(file_path):
	'''
	This function parses amazon data. It retrieves the reviews and good/bad label.

	:param file_path: file_path: path to traing/testing files
			-./data/test.ft.txt
			-./data/train.ft.txt
	:return:
		a sparse matrix w/ each row being a document and coluumn being a word
	'''

	count_vect = CountVectorizer(stop_words='english')

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

	X_train_counts = count_vect.fit_transform(review_str_list)

	return (X_train_counts, labels_list)

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


def parse_imdb_data_sklearn(file_path):
	'''

	:param file_path: file_path: path to training/testing files
			-./data/aclImdb/train
			-./data/aclImdb/test
	:return:
		a sparse matrix w/ each row being a document and coluumn being a word
	'''
	count_vect = CountVectorizer(stop_words='english')

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

	X_train_counts = count_vect.fit_transform(review_str_list)

	return (X_train_counts, labels_list)

if __name__ == "__main__":
	training_amazon = "./data/train.ft.txt"
	testing_amazon = "./data/test.ft.txt"
	training_imdb = "./data/aclImdb/train"
	testing_imdb = "./data/aclImdb/test"

	# a, b = parse_amazon_data_sklearn(testing_amazon)
	# c, d = parse_imdb_data_sklearn(testing_imdb)
	# print(a[0])