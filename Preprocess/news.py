#coding=utf-8
import re
from auxi import get_max_len, save_obj, load_obj, get_shuffle, get_item_by_idx
import numpy as np
import time
import csv

from collections import defaultdict
import linecache
import platform
if platform.platform().find('Microsoft') == -1:
	prefix = '/home/shzhou/Source/'
else:
	prefix = '/mnt/d/NLP/Source/'

alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
dataset = 'AGnews'

train_path = prefix + dataset + '/train.csv'
test_path = prefix + dataset + '/test.csv'

pre_train = prefix + dataset + '/pre_train.txt'
pre_test = prefix + dataset + '/pre_test.txt'

label_train = prefix + dataset + '/label_train.txt'
label_test = prefix + dataset + '/label_test.txt'

tmp_train = prefix + dataset + '/tmp_train.txt'

width = {'AGnews':704, 'dbpedia':704, 'yahoo_answers':4352, 'yelp_review':4864}
n_classes = {'AGnews':4, 'dbpedia':14, 'yahoo_answers':10, 'yelp_review':5}
train_num = {'AGnews':120000, 'dbpedia':560000, 'yahoo_answers':1400000, 'yelp_review':650000}

in_width = width[dataset]

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def count_len(fname):
	count = 4864
	num = 0
	total = 0
	with open(fname, 'r') as f:
		lines = f.readlines()
		for line in lines:
			line = line.split()
			if len(line) > count:
				num += 1
			total += 1
	print num
	print total

# position type is str in (alpha, position) pair
def create_dict():
	dic = defaultdict(int)
	i = 1
	for s in alphabet:
		dic[s] = str(i)
		i += 1
	return dic

def _load(fname, flag, in_width, shuffle = None):
	with open(fname, 'r') as fp:
		if shuffle is not None:
			shuffle.sort()
			res = [line for i, line in enumerate(fp) if i in shuffle]
		else:
			res = fp.readlines()
		if flag == 'data':
			return res	
		else:
			return list(map(int,res))

def load(pref, labelf, in_width, shuffle = None):
	return _load(pref, 'data', in_width, shuffle), _load(labelf, 'label', 0, shuffle)

def load_train(batch_size, in_width):
	shuffle = get_shuffle(0, train_num[dataset], batch_size) 
	return load(pre_train, label_train, in_width, shuffle)

def load_all_train(in_width):
	return load(pre_train, label_train, in_width)

def load_test(dummy, in_width):
	return load(pre_test, label_test, in_width)

def pad(pref):
	dic = create_dict()
	allsentence = []
	with open(pref, 'r+') as f:
		lines = f.readlines()
		for line in lines:
			line = line.strip().split()
			lineidx = [idx for idx in line]
			if len(lineidx) > in_width:
				lineidx = lineidx[:in_width]
			else:
				pad = [dic[' ']] * (in_width - len(lineidx))
				lineidx.extend(pad)
			lineidx = ' '.join(lineidx)
			allsentence.append(lineidx)
		allsentence = '\n'.join(allsentence)
		f.seek(0)
		f.write(allsentence)

def preprocess_dbpedia():
	with open(test_path, 'rb') as csvfile:
		reader = csv.reader(csvfile)
		labels = []
		contents = []
		for line in reader:
			labels.append(line[0].strip())
			content = remove_nonEng_bracket(line[1].strip())
			content += ' ' + remove_nonEng_bracket(line[2].strip())
			contents.append(content)
		labels = '\n'.join(labels)
		contents = '\n'.join(contents)
		with open(pre_test,'w') as w1:
			w1.write(contents)
		with open(label_test, 'w') as w2:
			w2.write(labels)

def preprocess(path, pre_path, label_path):
	with open(path, 'rb') as csvfile:
		reader = csv.reader(csvfile)
		labels = []
		contents = []
		for line in reader:
			col = len(line) - 1
			line = [remove_nonEng(l.strip()) for l in line]
			cls = int(line[0]) - 1
			labels.append(str(cls))
			content = ' '.join(line[1:])
			contents.append(content)
		labels = '\n'.join(labels)
		contents = '\n'.join(contents)
		with open(pre_path,'w') as w1:
			w1.write(contents)
		with open(label_path, 'w') as w2:
			w2.write(labels)

def remove_nonEng(s):
	return s.decode("ascii", errors = 'ignore').encode()

def remove_nonEng_bracket(s):
	pattern = re.compile(r'(\(.*?\))')
	finds = re.findall(pattern, s)
	for find in finds:
		if not test_Eng(find):
			s = s.replace(find, '')
	return s	

def test_Eng(s):
	try:
		s.decode("ascii").encode()
	except:
		return False
	else:
		return True

if __name__ == "__main__":
	pass
	preprocess(train_path, pre_train, label_train)
	preprocess(test_path, pre_test, label_test)
	# pad(pre_train)
	# pad(pre_test)
	# count_len(pre_train)
	# count_len(pre_test)
