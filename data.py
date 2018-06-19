#!/usr/bin/python
#_*_coding:utf-8_*_


'''
data process
'''


import sys
import time


class Data():
	def __init__(self, train_path, train_doc_path, test_path, test_doc_path):
		"""
		parameter:
			train_path: train_set
			train_doc_path: train_nid_data
			test_path: test_set
			test_doc_path: test_nid_data
		"""
		self.train_path = train_path
		self.test_path = test_path

	def covert_train(self, output_path): 
		"""
		parameter:
			output_path: train_nid_data_small
		"""
		print('covert_train')
	
	def filter_train(self, output_path): 
		""" filter the unclick data
		parameter:
		"""
		print('filer unclick')


def timestamp2date(ts):
	"""
	"""
	import time
	lt = time.localtime(ts)
	format_ = "%Y-%m-%d %H:%M:%S"
	return time.strftime(format_, lt)


def filter_data(txt_input, txt_output):
	print("------filter data----------")
	ofile = open(txt_output, 'w')
	with open(txt_input) as ifile:
		for line in ifile:
			line = line.strip('\r\n')
			uid, docid, time, click_flag, label = line.split('\t')
			if click_flag == '0' or label == '0':
				continue
			ofile.write('\t'.join([uid, docid, label]))
			ofile.write('\n')
	ofile.close()
	

def filter_data_parse_time(txt_input, txt_output):
	print("------filter_data_parse_time----------")
	ofile = open(txt_output, 'w')
	with open(txt_input) as ifile:
		for line in ifile:
			line = line.strip('\r\n')
			uid, docid, time, click_flag, label = line.split('\t')
			if click_flag == '0' or label == '0':
				continue
			ofile.write('\t'.join([uid, docid, timestamp2date(int(time)), label]))
			ofile.write('\n')
	ofile.close()

def filter_data_keep_time(txt_input, txt_output):
	print("------filter_data_keep_time----------")
	ofile = open(txt_output, 'w')
	with open(txt_input) as ifile:
		for line in ifile:
			line = line.strip('\r\n')
			uid, docid, time, click_flag, label = line.split('\t')
			if click_flag == '0' or label == '0':
				continue
			ofile.write('\t'.join([uid, docid, time, label]))
			ofile.write('\n')
	ofile.close()

def parse_data(txt_input, txt_output):
	print("------parse data----------")
	ofile = open(txt_output, 'w')
	with open(txt_input) as ifile:
		for line in ifile:
			line = line.strip('\r\n')
			cols = line.split('\t')
			cols[7] = str(len(cols[7].decode('utf-8')))
			ofile.write('\t'.join(cols))
			ofile.write('\n')
	ofile.close()

if __name__ == '__main__':
	if len(sys.argv) != 4:
		print("Usage: %s fun_name txt_input txt_output" % sys.argv[0])
		sys.exit(-1)
	fun = sys.argv[1]
	if fun in ['filter_data', 'parse_data', 'filter_data_parse_time', 'filter_data_keep_time']:
		fun = eval(fun)
		txt_input = sys.argv[2]
		txt_output = sys.argv[3]
		fun(txt_input, txt_output)
	elif True:
		pass
