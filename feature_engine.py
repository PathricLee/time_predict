#_*_coding:utf_*_

import numpy as np
import pandas as pd
import logging
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


logging.basicConfig(filename = os.path.join(os.getcwd(), 'log.txt'),\
        level = logging.DEBUG, format = '%(asctime)s - %(levelname)s: %(message)s')


class basic_extractor(object):
    def __init__(self, data, default_id = 0, input_dim = 1):
        self.item2id = {}
        self.default_id = default_id
        self.input_dim = input_dim
        self.build(data)
    def build(self, data):
        for input_ in data:
            for e in self.process(input_):
                if e not in self.item2id:
                    self.item2id[e] = len(self.item2id) + 1
    def input2id(self, data_):
        ret = []
        for input_ in data_:
            sub = [self.item2id[e] for e in self.process(input_) if e in self.item2id]
            ret.append(sub)
        return pad_sequences(ret, maxlen=self.input_dim, padding='post',\
                truncating='post', value=self.default_id)
    def process(self, input_):
        """ 处理每一行的数据，不同类型方法不一样"""
        return [input_]
    def get_feature_size(self):
        return len(self.item2id) + 1
    def get_input_dim(self):
        return self.input_dim


class keyword_extractor(basic_extractor):
    def build(self, data):
        ret = [self.process(input_) for input_ in data]
        tok = Tokenizer(num_words = 5000)
        tok.fit_on_texts(ret)

    def input2id(self. data):
        ret = [self.process(input_) for input_ in data]
        tok.texts_to_sequences(ret)
        return pad_sequences(ret, maxlen=self.input_dim, padding='post',\
                truncating='post', value=self.default_id)

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
        return []

class convertor(object):
    def __init__(self, used_fea, extracts, train_txt, test_txt, predict_txt):
        self.used_fea = used_fea
        self.extracts = extracts
        self.train_txt = train_txt
        self.test_txt = test_txt
        self.predict_txt = predict_txt

    def run(self):
        self.train_txt2id()
        self.test_txt2id('test')
        self.test_txt2id('online')

    def train_txt2id(self):
        df = pd.read_csv(self.train_txt, sep='\t')
        for i, fea in enumerate(self.used_fea):
            lst = list(df[fea])
            if fea == 'key_word': 
                ext = keyword_extractor(lst, 0, 10)
            else:
                ext = basic_extractor(lst, 0, 1)

            array_id = ext.input2id(lst)
            np.savetxt('train/feature/train_%s.txt' % fea, array_id, fmt="%d")
            self.extracts.append(ext)
            print(fea, ext.get_feature_size(), ext.get_input_dim())

    def test_txt2id(self, name):
        fname = self.test_txt if name == 'test' else self.predict_txt
        df = pd.read_csv(fname, sep='\t')
        for i, fea in enumerate(self.used_fea):
            lst = list(df[fea])
            ext = self.extracts[i]
            array_id = ext.input2id(lst)
            np.savetxt('train/feature/%s_%s.txt' % (name, fea), array_id, fmt="%d")


if __name__ == '__main__':
    # train, test, online
    used_fea = ['cid','cate','tag1','tag2', 'author', 'key_word']
    extracts = []
    train_txt = 'train/data/train.csv'
    test_txt = 'train/data/test.csv'
    online_txt = '../test/merge.csv'
    conv = convertor(used_fea, extracts, train_txt, test_txt, online_txt)
    conv.run()
