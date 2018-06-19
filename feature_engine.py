#_*_coding:utf_*_

import numpy as np


class basic_extractor(object):
    def __init__(self, data, default_id = 0, input_dim = 1):
        self.iddict = {}
        self.default_id = default_id
        self.input_dim = input_dim
        self.build(data)
    def build(self, data):
        for input_ in data:
            for e in self.process(input_):
                if e not in self.iddict:
                    self.iddict[e] = len(self.didict) + 1
    def input2id(self, data_):
        ret = []
        for input_ in data:
            sub = [self.iddict.get(e, self.default_id) for e in self.process(input_)]
        return np.array(ret)
    def process(self, input_):
        """ 处理每一行的数据，不同类型方法不一样"""
        ret = [input_.strip()]
        return ret
    def get_feature_size(self):
        return len(self.iddict) + 1
    def get_input_dim(self):
        return self.input_dim

class keyword_extractor(basic_extractor):
    def build(self, data):
        for e in data:
            if e not in self.iddict:
                self.iddict[e] = len(self.didict) + 1
        return input2id(data)
    def input2id(self, input_):
        ret = [self.iddict.get(e, self.default_id) for e in self.input_]
        ret = np.array(ret)[:, np.newaxis]
        return ret
    def get_feature_size(self):
        return len(self.iddict) + 1
    def process(self, keyword):
        keyword = keyword.strip('"{}')
        if keyword is not None and len(keyword)>1:
            try:
                words = [(''.join(item.split(':')[:-1]).strip('"'), int(item.split(':')[-1]))\
                        for item in keyword.split(',') ]
            except ValueError:
                print(keyword)
                continue
            sorted_words =  sorted(words, key = lambda x: x[1], reverse = True)

if __name__ == '__main__':
    # train, test, online
    used_fea = ['cid','cate','tag1','tag2', 'key_word', 'author', 'title']
    extracts = []
    # train 
    train = pd.read_csv('train/data/train.csv', sep='\t')
    for fea in used_fea:
        if fea == 'key_word': 
        elif fea in ['title']:
        else:
            ext = extractor()
            list_id = ext.build(list(all_data[ele]))
            extracts.append(ext)
    # test
    test = pd.read_csv('train/data/test.csv', sep='\t')
    for fea in used_fea:
        if fea == 'key_word': 
        elif fea in ['title']:
        else:
            ext = extracts[i]
            list_id = ext.build(list(all_data[ele]))
            extracts.append(ext)
    # online
    online = pd.read_csv('test/merge.csv', sep='\t')

