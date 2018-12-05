import os

def parse_amazon_data(file_path):
	'''
	This function parses amazon data. It retrieves the reviews and good/bad label.

	Input:
		file_path: path to traing/testing files
			-./data/test.ft.txt
			-./data/train.ft.txt

	Output:
		a tuple consisting of 
			1. a list of reviews (a big blob of string)
			2. a list of good/bad labels (stored as ints. 0 = bad, 1 = good)
	'''
	review_str_list = []
	labels_list = []

	with open(file_path, 'r') as f:
		for line in f:
			label_str, review_str = line.split(' ', 1)
			label = int(label_str[-1]) - 1		# original data is labeled with 1 and 2. change to labels of 0 and 1
		  
			# Not too sure what the best DS is here but gonna store everything in list for now
			review_str_list.append(review_str)
			labels_list.append(labels_list)

	return (review_str_list, labels_list)

def parse_imdb_data(file_path):
	'''
	This function parses amazon data. It retrives the reviews and good/bad label.

	Input:
		file_path: path to training/testing files
			-./data/aclImdb/train
			-./data/aclImdb/test
	
	Output:
		a tuple consisting of
			1. a list of reviews (a big blob of string)
			2. a list of good/bad labels (stored as ints. 0 = bad, 1 = good)
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
			break

		for file_name in os.listdir(file_path + "/" + folder_name):
			with open(file_path + "/" + folder_name + "/" + file_name, 'r', errors='ignore') as f:
				review_str = ""
				for line in f:
					review_str += line
			review_str_list.append(review_str)
			labels_list.append(label)

	return (review_str_list, labels_list)

