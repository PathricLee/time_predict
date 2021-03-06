#_*_coidng:utf-8_*_


"""
build model
"""

import sys
import time
import os

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
from sklearn.metrics import roc_auc_score

from keras_plot import LossHistory

def define_embedding(EMBEDDING_DIM, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, wei, name):
    """
        input_dim: repre the one hot length. how many words.
        output_dim: repre the embedding length
        input_length: how many words a time. often the max length
        wei: the initial weight
    """
    if wei is not None:
        return Embedding(output_dim=EMBEDDING_DIM, input_dim=MAX_NB_WORDS, input_length=MAX_SEQUENCE_LENGTH, weights=[wei], name=name)
    else:
        return Embedding(output_dim=EMBEDDING_DIM, input_dim=MAX_NB_WORDS, input_length=MAX_SEQUENCE_LENGTH, weights=wei, name=name)

def mean_bow(x, axis_):
    return K.mean(x, axis = axis_, keepdims = True)

def build_model(config, embed):
    """
        max_ele_len: size of feature space
    """
    #-----------------------model-------------------
    #uid
    uid_input = Input(shape=(config[0][1], ), dtype='int32')
    uid_embed = define_embedding(128, config[0][0]+1, config[0][1], embed[0], 'uid_embed')(uid_input)

    #cate
    cate_input = Input(shape=(config[1][1], ), dtype='int32')
    cate_embed = define_embedding(2, config[1][0]+1, config[1][1], embed[1], 'cate_embed')(cate_input)

    # tag1
    tag1_input = Input(shape=(config[2][1], ), dtype='int32')
    tag1_embed = define_embedding(16, config[2][0]+1, config[2][1], embed[2], 'tag1_embed')(tag1_input)

    # tag2
    tag2_input = Input(shape=(config[3][1], ), dtype='int32')
    tag2_embed = define_embedding(64, config[3][0]+1, config[3][1], embed[3], 'tag2_embed')(tag2_input)

    # author 
    author_input = Input(shape=(config[4][1], ), dtype='int32')
    author_embed = define_embedding(128, config[4][0]+1, config[4][1], embed[4], 'author_embed')(author_input)

    # key_word
    key_word_input = Input(shape=(config[5][1], ), dtype='int32')
    key_word_embed = define_embedding(100, config[5][0]+1, config[5][1], embed[5], 'key_word_embed')(key_word_input)
    lambda_lay = Lambda(mean_bow, output_shape=(1,100), arguments={'axis_':1})(key_word_embed)

    # title
    title_input = Input(shape=(config[6][1], ), dtype='int32')
    title_embed = define_embedding(100, config[6][0]+1, config[6][1], embed[6], 'title_embed')(title_input)
    lambda_title = Lambda(mean_bow, output_shape=(1,100), arguments={'axis_':1})(title_embed)

    # flatten
    f = Flatten()
    # activation
    a = Activation('softsign')

    # doc ensembel
    concat1 = concatenate([f(author_embed), f(cate_embed), f(tag1_embed), f(tag2_embed), f(lambda_lay)])
    drop1 = Dropout(0.2)(concat1)
    act1 = a(drop1)
    doc_den = Dense(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.01), name = 'doc_den')(act1)

    # conbine 
    concat2 = concatenate([f(uid_embed), doc_den, f(lambda_title)])
    act2 = a(concat2)
    den1 = Dense(units=30, activation='relu', kernel_regularizer=regularizers.l2(0.01), name = 'den1')(act2)
    den2 = Dense(units=1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01), name = 'den_sigmoid')(den1)

    model = Model(inputs=[uid_input, cate_input, tag1_input, tag2_input, author_input,\
            key_word_input, title_input], outputs=den2)

    model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc', 'mae'])

    model.summary()
    return model

def cal_acc(predict, y_test, fname):
    predict_label = np.argmax(predict, axis=1) + 1
    real_label = np.argmax(y_test, axis=1) + 1
    print(mean_absolute_error(predict_label, real_label))
    ids_, y_ = [x.reshape([x.shape[0], 1]) for x in [predict_label, real_label]]
    con_ = np.concatenate([ids_, y_], axis=1)
    df_ = pd.DataFrame(con_, columns = ['pred', 'real'], index = None)
    df_.to_csv(fname, index = False)

def cal_auc(y_true, y_pred, fname):
    #print(roc_auc_score(y_true, y_pred))
    y_pred = y_pred >=0.5
    y_pred = y_pred.astype(np.int32)
    y_true = y_true.astype(np.int32)
    equal = np.sum(y_pred == y_true)
    print('acc', float(equal)/y_true.shape[0])
    con_ = np.concatenate([y_true, y_pred], axis=1)
    np.savetxt(fname, con_, fmt="%d")
    

def load_feature(used_fea, name, load_label):
    ret = []
    for fea in used_fea:
        fname = 'train/feature/%s_%s.txt' % (name, fea)
        data = np.loadtxt(fname, dtype=np.int32, ndmin=2)
        print(data.shape)
        ret.append(data)
    if load_label:
        fname = 'train/feature/%s_%s.txt' % (name, 'label1')
        label = np.loadtxt(fname, dtype=np.int32, ndmin=2)
        return ret, label
    return ret

def load_embed(used_fea):
    ret = []
    for fea in used_fea:
        fname = 'train/embed/%s2vec' % (fea)
        if os.path.exists(fname):
            data = np.loadtxt(fname, dtype=np.float32, ndmin=2)
            print(fea, data.shape)
            ret.append(data)
        else:
            ret.append(None)
    return ret


if __name__ == '__main__':
    # ------------------------------data------------------------------------------
    # all train data
    used_fea = ['cid','cate','tag1','tag2', 'author', 'key_word', 'title']
    embed = load_embed(used_fea)
    config = np.loadtxt('config', dtype=np.int32)
    model = build_model(config, embed)

    print('load train')
    x_train, y_train = load_feature(used_fea, 'train', True)
    print('load test')
    x_test, y_test = load_feature(used_fea, 'test', True)
    #-------------------------model/train/test/save---------------------------------------

    #history
    history = LossHistory()
    # train
    model.fit(x_train, y_train, epochs=5, batch_size=2000,\
            validation_data=(x_test, y_test),\
            callbacks=[history])
    #model.save('20c.h5I1')
    model.save_weights('binary.weights1')
    model.save('model.now')

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


    history.loss_plot('epoch')
    history.loss_plot('batch')

    """
    model.load_weights('binary.weights', by_name=True)
    w = model.get_layer('title_embed').get_weights()
    for ws in w:
        print(ws.shape)
        np.savetxt('embed', ws)
    sys.exit(-1)
    """
    y = model.predict(x_test, verbose=0, batch_size=2000)
    y = y.reshape([y.shape[0], 1])
    print("Test mae")
    #cal_acc(y, y_test, 'predict_test.csv')
    cal_auc(y_test, y, 'predict_test.csv')

    #print("Train mae")
    #y_acc_train = model.predict(x_train, verbose=0, batch_size=516)
    #cal_acc(y_acc_train, y_train, 'predict_train,csv')

    print("Online predict")
    print('load online')
    x_online = load_feature(used_fea, 'online', False)
    y_predict = model.predict(x_online, verbose=0, batch_size=2000)
    y_pred = y_predict.reshape([y_predict.shape[0], 1])
    y_true = np.ones(shape=[y_predict.shape[0], 1])
    cal_auc(y_true, y_pred, 'online_test.csv')
    """
    y_predict_label = np.argmax(y_predict, axis=1) + 1
    raw_test_data = pd.read_csv('../test/merge.csv', sep='\t')
    y_index = raw_test_data[['cid', 'did']]
    # y_index['label'] = y_predict_label
    y_index.insert(y_index.shape[1], 'label', y_predict_label)
    y_index.to_csv('submit.csv', sep='\t', index = False, header=False)
    """

    del model
