#_*_coding:utf_*_

import numpy as np


class basic_extractor(object):
    def __init__(self, default_id = 0, default_len = 1):
        self.iddict = {}
        self.default_id = default_id
        self.input_dim = default_len
    def build(self, data):
        for e in data:
            if e not in self.iddict:
                self.iddict[e] = len(self.didict) + 1
        return self.input2id(data)
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
    # all train data
    all_data = pd.read_csv('train/merge.csv', sep='\t')
    all_data, train_num = split_train_test(all_data)
    train_data_len = all_data.shape[0]

    used_fea = ['cid','cate','tag1','tag2', 'key_word', 'author', 'title']
    data_train, data_eval, exts = [], [], [], []
    use_all_data = False
    # feature id-ing 
    train_index_split = train_num
    if use_all_data:
        train_index_split = train_data_len

    for ele in used_fea:
        if ele == 'key_word':
            array_id, word2id = keyword2idx(list(all_data[ele]))
            data_train.append(array_id[:train_index_split])
            data_eval.append(array_id[train_num:])
        elif ele in ['title']:
            array_id, tokenizer = title2id(list(all_data[ele]), None, 5000, 80)
            data_train.append(array_id[:train_index_split])
            data_eval.append(array_id[train_num:])
            word2id = get_word2id(tokenizer)
        else:
            ext = extractor()
            list_id = ext.build(list(all_data[ele]))
            exts.append(ext)
        print(ele, ext.get_feature_size())
        max_ele_len.append(ext.get_feature_size())
    # label
    labels = all_data['label'].astype(int).values - 1
    labels = to_categorical(np.asarray(labels))
    y_train = labels[:train_index_split]
    y_eval = labels[train_num:]
    print('data_train_shape:')
    for x in data_train:
        print(x.shape)

    data_predict = []
    raw_test_data = pd.read_csv('test/merge.csv', sep='\t')
    for i, fea in enumerate(used_fea):
        if fea == 'key_word': 
            fea_array = keyword2idx_dict(list(raw_test_data[fea]), word2id_lst[i])
            data_predict.append(fea_array)
        elif fea in ['title']:
            array_id = title2id(list(raw_test_data[fea]), word2id_lst[i])
            data_predict.append(array_id[:train_index_split])
        else:
            fea_array = np.array(transidx(list(raw_test_data[fea]), word2id_lst[i]))
            data_predict.append(fea_array[:, np.newaxis])
    print('data_predict_shape:')
    for x in data_predict:
        print(x.shape)
