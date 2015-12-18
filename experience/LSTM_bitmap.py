import numpy as np
import keras
import pickle
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Masking, TimeDistributedDense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM

alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

X = pickle.load(open('../training_x', 'rb'))
# X_bitmap = np.zeros((X.shape[0], X.shape[1], 54), dtype=np.bool)
# for i in range(X.shape[0]):
#     for j in range(X.shape[1]):
#         X_bitmap[i][j][X[i][j]] = 1
# print(X_bitmap.shape)
# print(X_bitmap[1])
#
# np.save('X_bitmap.dat', X_bitmap)
X_bitmap = np.load('X_bitmap.dat.npy')
y = pickle.load(open('../training_y', 'rb'))
all_y = np.array([np_utils.to_categorical(sample, 2) for sample in y])
print(all_y.shape)
print('loaded')
print()
print(X.shape, y.shape)
model = Sequential()
# model.add(Embedding(54, 32, input_length=200))
model.add(LSTM(100, return_sequences=True, input_shape=(200, 54)))
model.add(Dropout(0.5))
model.add(TimeDistributedDense(2))
model.add(Activation('sigmoid'))
print('compile')
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              class_mode="binary")
print("Train...")
model.fit(X_bitmap, all_y, nb_epoch=3,
          validation_split=0.2, show_accuracy=True)
score, acc = model.evaluate(X_bitmap, all_y,
                            batch_size=32,
                            show_accuracy=True)

model.save_weights('model')
