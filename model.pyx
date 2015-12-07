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

'''
    Train a LSTM on the IMDB sentiment classification task.
    The dataset is actually too small for LSTM to be of any advantage
    compared to simpler, much faster methods such as TF-IDF+LogReg.
    Notes:
    - RNNs are tricky. Choice of batch size is important,
    choice of loss and optimizer is critical, etc.
    Some configurations won't converge.
    - LSTM loss decrease patterns during training can be quite different
    from what you see with CNNs/MLPs/etc.
    GPU command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python imdb_lstm.py
'''

max_features = 20000
# cut texts after this number of words (among top max_features most common
# words)
maxlen = 10
batch_size = 32
print("Loading data...")
#(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features,
#                                                      test_split=0.2)

w2v = gensim.models.Word2Vec.load('word2vec.dat')


all_x = pickle.load(open('training_allx1.dat', 'rb'))
all_y = pickle.load(open('training_ally1.dat', 'rb'))
window_size = 5

embeddings = np.zeros((len(all_x), 5, 10))

for i, x in enumerate(all_x):
    for j, k in enumerate(all_x[i]):
        embeddings[i][j] = w2v[k]

print("Pad sequences (samples x time)")
all_y = np.array(all_y)
train_x = embeddings[:len(embeddings)*0.8]
train_y = all_y[:len(embeddings)*0.8]
test_x = embeddings[len(embeddings)*0.8:]
test_y = all_y[len(embeddings)*0.8:]
train_y, test_y = [np_utils.to_categorical(x, 4) for x in (train_y, test_y)]

print('generate model')
model = Sequential()
model.add(LSTM(128, input_shape=(5, 10)))
model.add(Dropout(0.5))
model.add(Dense(4))
model.add(Activation('softmax'))
print('compile')
# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              class_mode="categorical")

print("Train...")
model.fit(train_x, train_y, batch_size=batch_size, nb_epoch=2,
          validation_data=(test_x, test_y),
          show_accuracy=True)
score, acc = model.evaluate(test_x, test_y,
                            batch_size=batch_size,
                            show_accuracy=True)


print('Test score:', score)
print('Test accuracy:', acc)
