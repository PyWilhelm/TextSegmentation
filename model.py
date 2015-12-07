from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Masking
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.datasets import imdb

import gensim.models

import pickle


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
maxlen = 5
batch_size = 32
print("Loading data...")
#(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features,
#                                                      test_split=0.2)

print(gensim.models)
raw_x = pickle.load(open('training_x.dat', 'rb'))
print(raw_x[0])
raw_y = pickle.load(open('training_y.dat', 'rb'))

#model = gensim.models.Word2Vec(raw_x, workers=8)
#print(model)
#model.save('word2vec.dat')



all_x = pickle.load(open('training_allx1.dat', 'rb'))
all_y = pickle.load(open('training_ally1.dat', 'rb'))

all_x = [[ord(c) - ord('a') for c in x] for x in all_x]

print(len(all_x), len(all_y))





print("Pad sequences (samples x time)")
all_x = sequence.pad_sequences(all_x, maxlen=maxlen)
all_y = np.array(all_y)
train_x = all_x[:800]
train_y = all_y[:800]
test_x = all_x[800:1000]
test_y = all_y[800:1000]
y = np.array(all_y)

model = Sequential()
M = Masking(mask_value=0)
M._input_shape = train_x.shape
model.add(M)
model.add(LSTM(5))  # try using a GRU instead, for fun
model.add(Dropout(0.5))
model.add(Dense(128, 1))
model.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              class_mode="categorical")

print("Train...")
model.fit(train_x, train_y, batch_size=batch_size, nb_epoch=1,
          validation_data=(test_x, test_y),
          show_accuracy=True)
score, acc = model.evaluate(test_x, test_y,
                            batch_size=batch_size,
                            show_accuracy=True)


print('Test score:', score)
print('Test accuracy:', acc)
