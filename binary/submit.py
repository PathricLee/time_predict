#_*_coding:utf-8_*_
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from keras.layers import Input, Conv1D, MaxPool1D, Dense, Activation, Flatten, concatenate, Subtract, Multiply, Embedding
from keras import regularizers
from keras.models import Model
from keras import optimizers

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


from keras.models import load_model


#----------------model-------------------------------
model = load_model('20c.h5')
model.save('21c.h5')
model.summary()

#-------------------data-----------------------------

#data = pd.read_csv('test/merge.csv')

