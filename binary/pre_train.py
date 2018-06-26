#_*_coding_*_


from gensim.models import Word2Vec
import numpy as np

class char2vec(object):
    def __init__(self, word2vec, word2index, maxnum, size):
        self.word2vec = word2vec
        self.word2index = word2index
        self.maxnum = maxnum
        self.size = size
        self.w2i = []
        self.file2dict()
        self.run()

    def run(self):
        model = Word2Vec.load(self.word2vec)
        embed = np.zeros([self.maxnum + 1, self.size])
        vocab = model.wv.vocab
        for i, w in enumerate(self.w2i):
            index = i + 1
            if w in vocab:
                embed[index] = model[w]
        np.savetxt('train/embed/title2vec', embed, fmt="%f")


    def file2dict(self):
        with open(self.word2index, 'r', encoding='utf-8') as fin:
            for i, line in enumerate(fin):
                if i == self.maxnum:
                    break
                line = line.rstrip()
                word, index = line.split('\t')
                self.w2i.append(word)

if __name__ == '__main__':
    cv = char2vec('../time_predict/train/word2vec', 'train/feature/word2indx_title.txt', 5000, 100)
                
