#_*_coding:utf-8_*_

import pandas as pd
import numpy as np
from sklearn.utils import shuffle 
import time

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

def read_offline():
    data_train = pd.read_csv('train_set_use_keep_time', sep='\t', header=None,\
            names=['cid','did', 'time', 'label'])
    doc_info = pd.read_csv('train_nid_data_content_length',\
            sep='\t',\
            header=None,\
            names = ['did','cate','url','site','author','author_level',\
            'title','content_len','key_word','tag1','tag2','pic_num',\
            'pic_att1','pic_att2','pic_att3','pic_att4','pic_att5'])
    #### 现在需要讲两者json 进来
    merge = pd.merge(data_train, doc_info, on='did', how='left')
    merge.to_csv('merge.csv', index =False, sep='\t')

def read_online():
    data_train = pd.read_csv('test/test_set', sep='\t', header=None,\
            names=['cid','did'])
    doc_info = pd.read_csv('test/test_nid_data_len',\
            sep='\t',\
            header=None,\
            names = ['did','cate','url','site','author','author_level',\
            'title','content_len','key_word','tag1','tag2','pic_num',\
            'pic_att1','pic_att2','pic_att3','pic_att4','pic_att5'])

    #### 现在需要讲两者json 进来
    merge = pd.merge(data_train, doc_info, on='did', how='left')

    merge.to_csv('test/merge.csv', index =False, sep='\t')

def split_train_test():
    """
    """
    df = pd.read_csv('../train/merge.csv', sep='\t', na_values='\\N')
    df = df.fillna(' ')
    start  = '2018-05-02 00:00:00'
    end = '2018-05-02 23:59:59'
    s, e = [int(time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S"))) for x in [start, end]]
    train = df[df.time < s]
    train = shuffle(train)
    print("train shape: ", train.shape)
    test = df[df.time >= s]
    print("test shape: ", test.shape)
    #return pd.concat([train, test], axis = 0), train.shape[0]
    train.to_csv('train/data/train.csv', sep='\t', index = False, encoding='utf-8')
    test.to_csv('train/data/test.csv', sep='\t', index = False, encoding='utf-8')


if __name__ == '__main__':
    """
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
    """
    split_train_test()
