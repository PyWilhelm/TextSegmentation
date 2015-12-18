from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Masking, TimeDistributedDense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.datasets import imdb

import gensim.models

import pickle
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

max_features = 30

print("Loading data...")

w2v = gensim.models.Word2Vec.load('/Users/wilhelm/TextSegment/word2vec.dat')


all_x = pickle.load(open('/Users/wilhelm/TextSegment/training_x.dat', 'rb'))[:100]
all_y = pickle.load(open('/Users/wilhelm/TextSegment/training_y.dat', 'rb'))[:100]
print('loaded')
all_x = [[w2v[c] for c in sample] for sample in all_x]
max_length = max([len(sample) for sample in all_x])

window_size = 5

#all_x = sequence.pad_sequences(all_x)
#all_y = sequence.pad_sequences(all_y, value=4, dtype=np.int8)
#print(all_x.shape, all_y.shape)

all_y = [np_utils.to_categorical(sample, 5) for sample in all_y]



print('generate model')
model = Sequential()
model.add(LSTM(4, input_shape=(80000, 10)))
# model.add(Dropout(0.5))
# model.add(TimeDistributedDense(4))
# model.add(Activation('softmax'))
print('compile')
# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              class_mode="categorical")

print("Train...")
print(np.array([y[100] for y in all_y]).shape)
model.fit(all_x, [y[100] for y in all_y], validation_split=0.2,
          show_accuracy=True)
score, acc = model.evaluate(all_x[1000: 2000], all_y[1000: 2000],
                            show_accuracy=True)


print('Test score:', score)
print('Test accuracy:', acc)
