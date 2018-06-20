#_*_coding:utf_*_

import numpy as np
import pandas as pd
import logging
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical


logging.basicConfig(filename = os.path.join(os.getcwd(), 'log.txt'),\
        level = logging.DEBUG, format = '%(asctime)s - %(levelname)s: %(message)s')


class basic_extractor(object):
    def __init__(self, data, num_words = None, input_dim = 1, default_id = 0):
        self.default_id = default_id
        self.input_dim = input_dim
        self.num_words = num_words
        if self.num_words:
            self.tok = Tokenizer(num_words, filters=' ')
        else:
            self.tok = Tokenizer(filters = ' ')
        self.build(data)

    def build(self, data):
        texts = [self.process(input_) for input_ in data]
        self.tok.fit_on_texts(texts)
        if self.num_words is None:
            self.num_words = len(self.tok.word_index)

    def input2id(self, data):
        texts = [self.process(input_) for input_ in data]
        ret = self.tok.texts_to_sequences(texts)
        return pad_sequences(ret, maxlen=self.input_dim, padding='post',\
                truncating='post', value=self.default_id)
    def process(self, input_):
        """ 处理每一行的数据，不同类型方法不一样"""
        return str(input_)
    def get_feature_size(self):
        return self.num_words
    def get_input_dim(self):
        return self.input_dim


class keyword_extractor(basic_extractor):
    def process(self, input_):
        ret = []
        input_ = input_.strip('"{}')
        if input_ is not None and len(input_)>1:
            try:
                words = [(''.join(item.split(':')[:-1]).strip('"'), int(item.split(':')[-1]))\
                        for item in input_.split(',') ]
            except ValueError:
                logging.info(input_ + ":: has no keywrods")
            sorted_words =  sorted(words, key = lambda x: x[1], reverse = True)
            #ret = [w for w, v in sorted_words]
            ret = [w for w, v in words]
        return ' '.join(ret)


class title_extractor(basic_extractor):
    def process(self, input_):
        return ' '.join(list(input_))

class convertor(object):
    def __init__(self, used_fea, extracts, train_txt, test_txt, predict_txt, save_label = False):
        self.used_fea = used_fea
        self.extracts = extracts
        self.train_txt = train_txt
        self.test_txt = test_txt
        self.predict_txt = predict_txt
        self.save_label = save_label

    def run(self):
        self.train_txt2id()
        self.test_txt2id('test')
        self.save_label = False
        self.test_txt2id('online')
        self.save_config('train/config_add')

    def train_txt2id(self):
        df = pd.read_csv(self.train_txt, sep='\t')
        for i, fea in enumerate(self.used_fea):
            lst = list(df[fea])
            if fea == 'key_word': 
                ext = keyword_extractor(lst, 6000, 6, 0)
            elif fea == 'title':
                ext = title_extractor(lst, 5000, 50, 0) 
            else:
                ext = basic_extractor(lst, None, 1, 0)
            array_id = ext.input2id(lst)
            np.savetxt('train/feature/train_%s.txt' % fea, array_id, fmt="%d")
            self.extracts.append(ext)
            print(fea, ext.get_feature_size(), ext.get_input_dim())
            fwrite = open('train/feature/word2indx_%s.txt' % fea, 'w')
            for k, v in ext.tok.word_index.items():
                line = "%s\t%s\n" % (k, v)
                fwrite.write(line)
            fwrite.close()
        if self.save_label:
            self.write_label(df, 'train')

    def test_txt2id(self, name):
        fname = self.test_txt if name == 'test' else self.predict_txt
        df = pd.read_csv(fname, sep='\t')
        for i, fea in enumerate(self.used_fea):
            lst = list(df[fea])
            ext = self.extracts[i]
            array_id = ext.input2id(lst)
            np.savetxt('train/feature/%s_%s.txt' % (name, fea), array_id, fmt="%d")
        if self.save_label:
            self.write_label(df, name)

    def write_label(self, df, name):
        labels = df['label'].astype(int).values - 1
        labels = to_categorical(np.asarray(labels))
        np.savetxt('train/feature/%s_label.txt' % name, labels, fmt="%d")

    def save_config(self, fname):
        fwrite = open(fname, 'w')
        for ext in self.extracts:
            line = "%s\t%s\n" % (ext.get_feature_size(), ext.get_input_dim())
            fwrite.write(line)
        fwrite.close()


if __name__ == '__main__':
    # train, test, online
    #used_fea = ['cid','cate','tag1','tag2', 'author', 'key_word']
    used_fea = ['title']
    extracts = []
    train_txt = 'train/data/train.csv'
    test_txt = 'train/data/test.csv'
    online_txt = '../test/merge.csv'
    conv = convertor(used_fea, extracts, train_txt, test_txt, online_txt, False)
    conv.run()
