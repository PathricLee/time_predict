#_*_coidng:utf-8_*_


"""
build model
"""

import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from keras.layers import Input, Conv1D, MaxPool1D, Dense, Activation, Flatten, concatenate, Subtract, Multiply, Embedding, Lambda,Dropout
from keras import regularizers
from keras.models import Model
from keras import optimizers

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.utils.np_utils import to_categorical
from keras import backend as K

from sklearn.utils import shuffle 

from keras_plot import LossHistory

def define_embedding(EMBEDDING_DIM, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, wei = None):
    """
        input_dim: repre the one hot length. how many words.
        output_dim: repre the embedding length
        input_length: how many words a time. often the max length
        wei: the initial weight
    """
    return Embedding(output_dim=EMBEDDING_DIM, input_dim=MAX_NB_WORDS, input_length=MAX_SEQUENCE_LENGTH, weights=wei)

def mean_bow(x, axis_):
    return K.mean(x, axis = axis_, keepdims = True)

def set2idx(set_):
    """
    """
    uid2idx, idx2uid = [dict(), dict()]
    # must be set [0-len(set)]
    for i, x in enumerate(set_):
        uid2idx[x] = i + 1
        idx2uid[i + 1] = x
    return uid2idx, idx2uid

def transidx(list_, uid2idx):
    return [uid2idx.get(x, 0) for x in list_]

def list2idx(list_):
    set_ = set(list_)
    uid2idx, idx2uid = set2idx(set_)
    return transidx(list_, uid2idx), uid2idx

def keyword2idx_dict(list_, word2index):
    top = 10
    ret = np.zeros(shape=(len(list_), top))
    for i, keyword in enumerate(list_):
        keyword = keyword.strip('"{}')
        if keyword is not None and len(keyword)>1:
            try:
                words = [(''.join(item.split(':')[:-1]).strip('"'), int(item.split(':')[-1]))\
                        for item in keyword.split(',') ]
            except ValueError:
                print(keyword)
                continue
            sorted_words =  sorted(words, key = lambda x: x[1], reverse = True)
            for j, (key, value) in enumerate(sorted_words[:top]):
                ret[i][j] = word2index.get(key, 0)
    return ret

def keyword2idx(list_):
    word2index = {} # [0+[1-len()]]
    top = 10
    ret = np.zeros(shape=(len(list_), top))
    for i, keyword in enumerate(list_):
        keyword = keyword.strip('"{}')
        if keyword is not None and len(keyword)>1:
            try:
                words = [(''.join(item.split(':')[:-1]).strip('"'), int(item.split(':')[-1]))\
                        for item in keyword.split(',') ]
            except ValueError:
                print(keyword)
                continue
            sorted_words =  sorted(words, key = lambda x: x[1], reverse = True)
            for j, (key, value) in enumerate(sorted_words[:top]):
                if key not in word2index:
                    word2index[key] = len(word2index) + 1
                ret[i][j] = word2index.get(key, 0)
    return ret, word2index


def num2id(list_, fea):
    ret = np.zeros(shape=(len(list_), 1))
    if fea == 'pic_num': #51
        for i, line in enumerate(list_):
            try:
                num = int(list_.strip())
            except:
                num = 0
                pass
            if num < 50:
                ret[i][0] = num
            else:
                ret[i][0] = 50
    if fea == 'content_len': #26
        for i, line in enumerate(list_):
            try:
                num = int(list_.strip())
            except:
                num = 0
                pass
            if num < 1000:
                ret[i][0] = int(num / 100) #[0-9]
            elif num < 2000:
                ret[i][0] = 10 + int((num - 1000) / 200) #[10-14]
            elif num < 3500:
                ret[i][0] = 15 + int((num - 2000) / 300) #[15-19]
            elif num < 5500: 
                ret[i][0] = 20 + int((num - 3500) / 400) #[20-24]
            else:
                ret[i][0] = 25
    return ret


def split_train_test(df):
    """
    """
    start  = '2018-05-02 00:00:00'
    end = '2018-05-02 23:59:59'
    s, e = [int(time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S"))) for x in [start, end]]
    # because xx. s is enough
    train = df[df.time < s]
    train = shuffle(train)
    print("train shape: ", train.shape)
    test = df[df.time >= s]
    print("test shape: ", test.shape)
    return pd.concat([train, test], axis = 0), train.shape[0]

def build_model(max_ele_len):
    """
        max_ele_len: size of feature space
    """
    #-----------------------model-------------------
    #uid
    uid_input = Input(shape=(1, ), dtype='int32')
    uid_embed = define_embedding(128, max_ele_len[0], 1)(uid_input)

    #cate
    cate_input = Input(shape=(1, ), dtype='int32')
    cate_embed = define_embedding(2, max_ele_len[1], 1)(cate_input)

    # tag1
    tag1_input = Input(shape=(1, ), dtype='int32')
    tag1_embed = define_embedding(16, max_ele_len[2], 1)(tag1_input)

    # tag2
    tag2_input = Input(shape=(1, ), dtype='int32')
    tag2_embed = define_embedding(64, max_ele_len[3], 1)(tag2_input)

    # key_word
    key_word_input = Input(shape=(10, ), dtype='int32')
    key_word_embed = define_embedding(100, max_ele_len[4], 10)(key_word_input)
    print(key_word_embed.shape)
    lambda_lay = Lambda(mean_bow, output_shape=(1,100), arguments={'axis_':1})(key_word_embed)

    # author 
    author_input = Input(shape=(1, ), dtype='int32')
    author_embed = define_embedding(128, max_ele_len[5], 1)(author_input)

    # title
    title_input = Input(shape=(80, ), dtype='int32')
    title_embed = define_embedding(100, max_ele_len[6], 80)(title_input)
    lambda_title = Lambda(mean_bow, output_shape=(1,100), arguments={'axis_':1})(title_embed)

    # flatten
    f = Flatten()
    # activation
    a = Activation('softsign')

    # doc ensembel
    concat1 = concatenate([f(author_embed), f(cate_embed), f(tag1_embed), f(tag2_embed), f(lambda_lay)])
    drop1 = Dropout(0.2)(concat1)
    act1 = a(drop1)
    doc_den = Dense(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(act1)

    # conbine 
    concat2 = concatenate([f(uid_embed), doc_den, f(lambda_title)])
    act2 = a(concat2)
    den1 = Dense(units=30, activation='relu', kernel_regularizer=regularizers.l2(0.01))(act2)
    den2 = Dense(units=5, activation='softmax', kernel_regularizer=regularizers.l2(0.01))(den1)

    model = Model(inputs=[uid_input, cate_input, tag1_input, tag2_input, key_word_input,\
            author_input, title_input], outputs=den2)

    model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

    model.summary()
    return model

def cal_acc(predict, y_test):
    predict_label = np.argmax(predict, axis=1) + 1
    real_label = np.argmax(y_test, axis=1) + 1
    print(mean_absolute_error(predict_label, real_label))
    ids_, y_ = [x.reshape([x.shape[0], 1]) for x in [predict_label, real_label]]
    con_ = np.concatenate([ids_, y_], axis=1)
    df_ = pd.DataFrame(con_, columns = ['pred', 'real'], index = None)
    df_.to_csv('predict.csv', index = False)

def get_word2id(fea):
    word2id = {}
    if fea == 'content_len':#26
        word2id = {i:i for i in range(26)}
    if fea == 'pic_num': #51
        word2id = {i:i for i in range(51)}
    return word2id

def title2id(lst, tokenizer = None, MAX_NB_WORDS = None, MAX_SEQUENCE_LENGTH = None): 
    txt = []
    for line in lst:
        line = line.strip()
        line = ' '.join(line)
        chars = line.split()
        txt.append(chars)

    if tokenizer is None
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        tokenizer.fit_on_texts(txt)

        seq = tokenizer.texts_to_sequences(txt)

        seq_pad = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
        return seq_pad, tokenizer
    else:
        seq = tokenizer.texts_to_sequences(txt)
        seq_pad = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
        return seq_pad
        
        

        

if __name__ == '__main__':
    # ------------------------------data------------------------------------------
    # all train data
    all_data = pd.read_csv('train/merge.csv', sep='\t')
    all_data, train_num = split_train_test(all_data)
    train_data_len = all_data.shape[0]

    used_fea = ['cid','cate','tag1','tag2', 'key_word', 'author', 'title']
    data_train, data_eval, max_ele_len, word2id_lst = [], [], [], []
    use_all_data = False
    # feature id-ing 
    train_index_split = train_num
    if use_all_data:
        train_index_split = train_data_len
    # 关于这部分，不应该online predict的数据也加进来，真实模拟，我们只有训练集。
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
            list_id, word2id = list2idx(list(all_data[ele]))
            data_train.append(np.array(list_id[:train_index_split])[:, np.newaxis])
            data_eval.append(np.array(list_id[train_num:])[:, np.newaxis])
        print(ele, len(word2id))
        max_ele_len.append(len(word2id) + 1)
        word2id_lst.append(word2id)
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


    #-------------------------model/train/test/save---------------------------------------

    #history
    history = LossHistory()
    # train
    model = build_model(max_ele_len)
    model.fit(data_train, y_train, epochs=8, batch_size=516,\
            validation_data=(data_eval, y_eval),\
            callbacks=[history])

    score = model.evaluate(data_eval, y_eval, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    history.loss_plot('epoch')
    history.loss_plot('batch')

    y = model.predict(data_eval, verbose=0, batch_size=516)
    print("Test mae")
    cal_acc(y, y_eval)

    #mae  train
    print("Train mae")
    y_acc_train = model.predict(data_train, verbose=0, batch_size=516)
    cal_acc(y_acc_train, y_train)
    model.save('20c.h5I1')

    y_predict = model.predict(data_predict, verbose=0, batch_size=516)
    y_predict_label = np.argmax(y_predict, axis=1) + 1
    y_index = raw_test_data[['cid', 'did']]
    # y_index['label'] = y_predict_label
    y_index.insert(y_index.shape[1], 'label', y_predict_label)
    y_index.to_csv('submit.csv', sep='\t', index = False, header=False)

    del model
