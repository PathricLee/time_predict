#_*_coding:utf-8_*_


from gensim.models import Word2Vec
import sys
import os
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

import logging

logging.basicConfig(filename = os.path.join(os.getcwd(), 'log.txt'),\
        level = logging.DEBUG, format = '%(asctime)s - %(levelname)s: %(message)s')


def train():
    logging.info('read data')
    text = []
    txt = 'title'
    with open(txt, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            line = ' '.join(line)
            cols = line.split()
            text.append(cols)
    txt = '../test/title'
    with open(txt, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            line = ' '.join(line)
            cols = line.split()
            text.append(cols)
    logging.info('gensim')
    model = Word2Vec(text, sg=1, size=100,  window=5,\
                    min_count=5,  negative=3, sample=0.001, hs=1, workers=4)  
    logging.info('save model')
    fname = 'word2vec'
    model.save(fname)

def test():
    fname = 'word2vec'
    model = Word2Vec.load(fname)
    print(model.similarity('我', '你'))
    print(model.similarity('我', '红'))
    sim_words = model.similar_by_word('红')
    for k, v in sim_words:
        try:
            print(k, v)
        except:
            pass
    vocal = model.wv.vocab
    for w in vocal:
        print(w)

if __name__ == '__main__':
    train()

