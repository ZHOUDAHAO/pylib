import cPickle as pickle
import random
import no_warnings
import numpy as np
from collections import Counter

def add_pkl2path(path):
	if not path.endswith('.pkl'):
		path = path + '.pkl'
	return path

def save_obj(obj, path):
	path = add_pkl2path(path)
	with open(path, 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path):
	path = add_pkl2path(path)
	with open(path, 'rb') as f:
		return pickle.load(f)

def softmax(x):
	x = x - x.max(axis = 1)[:,np.newaxis]
	sumx = np.sum(np.exp(x), axis = 1)
	return np.exp(x)/sumx[:,np.newaxis]

def calc_precision(pred, labels):
	p = np.argmax(pred, axis = 1)
	l = np.argmax(labels, axis = 1)
	rate = np.sum(p == l,dtype = np.float32)/len(l)
	return rate

def drop_empty_lines(prepro_path, file_path, start = 0):
	with open(prepro_path, 'w') as w:
		with open(file_path,'r') as f:
			lines = f.readlines()
			for line in lines[start:]:
				line = line.strip()
				if not line:
					continue
				w.write(line)
				w.write('\n')

def random_init(rate, shape):
	np.random.seed(41)
	return rate * np.random.rand(*shape)

def get_uni_bitoken(x,y): # unique bitoken
	if x > y:
		return x + ' ' + y
	else:
		return y + ' ' + x
		
def get_bitoken(x,y):
	return x + ' ' + y

def get_words(token):
	return token.split()[0],token.split()[1]

# get sentences contain only English words
def get_sentences(file_path, start, end):
	sentences = []
	with open(file_path,'r') as fi:
		f = fi.readlines()
		for line in f:
			if end == 0:
				words = line.split()[start:]
			else:
				words = line.split()[start:end]
			sentences.append(words)
	return sentences
				
# start means where English words start, end is where English words end
def get_occur_time(sentences): 
	occur_time = Counter()
	for words in sentences:
		for word in words:
			occur_time[word] += 1
	return occur_time

def get_embedding(sentences):
	embedding = {}
	word_num = 0
	for words in sentences:
		for word in words:
			if word not in embedding:
				embedding[word] = word_num
				word_num += 1
	return embedding, word_num

def get_cooccur_time(sentences, context_size):
	cooccur_time = Counter()
	for words in sentences:
		for i in range(len(words) - 1):			
			minpos = max(0, i - context_size - 1)
			maxpos = min(len(words), i + context_size + 1)
			for j in range(minpos, maxpos):
				if i == j:
					continue
				token = (words[i], words[j])						
				cooccur_time[token] += 1
	return cooccur_time

# assume the last element of each line is the tag(class, etc.)
def get_tag(file_path):
	with open(file_path,'r') as f:
		return [int(line.split()[-1]) for line in f.readlines()]

def get_labels(tag, n_classes):
	labels = np.zeros((len(tag), n_classes),dtype = np.float32)	
	labels[np.arange(labels.shape[0]),tag] = 1.0
	return labels

# print multiple empty lines in order to separate different outputs
def print_separator():
	for _ in range(10):
		print

def get_shuffle(a, b, k):
	return random.sample(range(a,b), k)

def attract():
	print '*' * 80

def get_max_len(sentences): # get the max length of a sentence for padding
	return max([len(words) for words in sentences])

def get_item_by_idx(a, List):
	return [a[x] for x in List]
