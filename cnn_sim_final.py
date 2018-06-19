# coding: utf-8

print(" ----------------import----------------------------")
import numpy as np
import pandas as pd

from keras.layers import Input, Conv1D, MaxPool1D, Dense, Activation, Flatten, concatenate, Subtract, Multiply, Embedding
from keras import regularizers
from keras.models import Model
from keras import optimizers

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


print("---------------config-------------------------------")
data = './data/'
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1


print("---------------------data------------------------------")
print("1: read data and validate")
data = pd.read_csv('./data/train_pre.csv', na_values='NULL')
data.head()
data = data.dropna()
data.info()

print("2: parse what you care")
data_q1 = data['question1'].values.tolist()
data_q2 = data['question2'].values.tolist()
data_q1[0:10]
label = data['is_duplicate'].values
label[0:10]

print("3: define tokenizer and sequenz the strs list")
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(data_q1+data_q2)

seq1 = tokenizer.texts_to_sequences(data_q1)
seq2 = tokenizer.texts_to_sequences(data_q2)

print("4: padding it to a fix length")
seq1_pad = pad_sequences(seq1, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
seq2_pad = pad_sequences(seq2, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

print("5: shuffle and split train & test data")
indices = np.arange(seq1_pad.shape[0])
np.random.shuffle(indices)
seq1_pad_suf = seq1_pad[indices]
seq2_pad_suf = seq2_pad[indices]
label_suf = label[indices]

nb_validation_samples = int(VALIDATION_SPLIT * seq2_pad_suf.shape[0])

seq1_train = seq1_pad_suf[:-nb_validation_samples]
seq2_train = seq2_pad_suf[:-nb_validation_samples]
y_train = label_suf[:-nb_validation_samples]

seq1_test = seq1_pad_suf[-nb_validation_samples:]
seq2_test = seq2_pad_suf[-nb_validation_samples:]
y_test = label_suf[-nb_validation_samples:]

print("6: the almost same way to online test data")
dtest = pd.read_csv('./data/test_pre.csv', na_values='NULL')
dtest['question2'] = dtest['question2'].astype(str)
dtest['question1'] = dtest['question1'].astype(str)

t_data_q1 = dtest['question1'].values.tolist()
t_data_q2 = dtest['question2'].values.tolist()
t_ids = dtest['test_id'].astype(int).values

t_seq_q1 = tokenizer.texts_to_sequences(t_data_q1)
t_seq_q2 = tokenizer.texts_to_sequences(t_data_q2)

t_seq_q1_pad = pad_sequences(t_seq_q1, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
t_seq_q2_pad = pad_sequences(t_seq_q2, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

print("7: if needed, load the pretrained embedding network")
embedding_index = {}
for line in open('data/glove.6B.%sd.txt' % EMBEDDING_DIM):
	line = line.strip()
	segs = line.split()
	word = segs[0]
	vec = np.array(segs[1:], dtype='float32')
	embedding_index[word] = vec

init_embed = np.zeros((MAX_NB_WORDS + 1, EMBEDDING_DIM))

for word, index in tokenizer.word_index.items():
	if index <= MAX_NB_WORDS and word in embedding_index:
		init_embed[index] = embedding_index[word]


# ---------------------model------------------------------
qus1 = Input(shape=(MAX_SEQUENCE_LENGTH, ), dtype='int32')
qus2 = Input(shape=(MAX_SEQUENCE_LENGTH, ), dtype='int32')

embedding = Embedding(output_dim=EMBEDDING_DIM, input_dim=MAX_NB_WORDS+1, input_length=MAX_SEQUENCE_LENGTH, weights=[init_embed])
x1 = embedding(qus1)
x2 = embedding(qus2)

conv = Conv1D(64, 4, activation='tanh')
c1 = conv(x1)
c2 = conv(x2)

maxp = MaxPool1D(pool_size=4)
m1 = maxp(c1)
m2 = maxp(c2)

fla = Flatten()
f1 = fla(m1)
f2 = fla(m2)

sub = Subtract()([f1, f2])
mul = Multiply()([f1, f2])
merge = concatenate([f1, f2, sub, mul])

den0 = Dense(units=120, activation='relu', kernel_regularizer=regularizers.l2(0.01))(merge)
den1 = Dense(units=30, activation='relu', kernel_regularizer=regularizers.l2(0.01))(den0)
den2 = Dense(units=1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))(den1)

model = Model(inputs=[qus1, qus2], outputs=den2)

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['binary_accuracy'])

model.fit([seq1_train, seq2_train], y_train, epochs=10, batch_size=200)

score = model.evaluate([seq1_train, seq2_train], y_train, verbose=0) 
print('train score:', score[0])
print('train accuracy:', score[1])
score = model.evaluate([seq1_test, seq2_test], y_test, verbose=0) 
print('Test score:', score[0])
print('Test accuracy:', score[1])

y = model.predict([t_seq_q1_pad, t_seq_q2_pad], verbose=0)
ids_, y_ = [x.reshape([x.shape[0], 1]) for x in [t_ids, y]]
con_ = np.concatenate([ids_, y_], axis=1)
df_ = pd.DataFrame(con_, columns = ['test_id', 'is_duplicate'], index = None)
df_['test_id'] = df_['test_id'].astype(int)
df_.to_csv('predict.csv', index = False)

#p_y = pd.DataFrame(y)
#p_y.to_csv('predict.csv')
#p_y.to_csv('predict.csv', header=['test_id', 'is_duplicate'])
#model.save('20c.h5')
model.summary()
del model

