import numpy as np
import keras
import pickle
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Masking, TimeDistributedDense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
import sys


def get_model():
    model = Sequential()
    #model.add(Embedding(54, 32, input_length=200))
    model.add(LSTM(128, return_sequences=True, input_shape=(200, 54*5)))
    model.add(Dropout(0.2))
    model.add(TimeDistributedDense(2))
    model.add(Activation('sigmoid'))
    print('compile')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  class_mode="binary")
    return model


def prepare_data(x_filename, y_filename, size=10000):
    X = pickle.load(open(x_filename, 'rb'))
    print(X.shape)
    X_bitmap = np.zeros((size, X.shape[1], 54*5), dtype=np.bool)
    y = pickle.load(open(y_filename, 'rb'))
    all_y = np.array([np_utils.to_categorical(sample, 2) for sample in y])
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            range_start = -2 if j >= 2 else -j
            range_end = 3 if j <= X.shape[1] - 3 else X.shape[1] - j
            for k in range(range_start, range_end):
                index = X[i][j + k] + 54 * (k + 2)
                X_bitmap[i % size][j][index] = 1

        if (i % size) == (size - 1):
            yield X_bitmap, all_y[i - size + 1: i + 1]
            X_bitmap = np.zeros((size, X.shape[1], 54*5), dtype=np.bool)

    if np.max(X_bitmap) > 0:
        yield X_bitmap[: X.shape[0] % size], all_y[X.shape[0] - X.shape[0] % size:]


if __name__ == '__main__':

    train_x_file = '../' + sys.argv[1]
    train_y_file = '../' + sys.argv[2]
    modelfile = sys.argv[3]
    # test_x_file = '../' + sys.argv[3]
    # test_y_file = '../' + sys.argv[4]


    print('loaded')
    model = get_model()
    print("Train...")
    for epoch in range(2):
        iterator = prepare_data(train_x_file, train_y_file)
        print('epoch:', epoch)
        for x, y in iterator:
            model.fit(x,y, nb_epoch=1, batch_size=32, show_accuracy=True)
    model.save_weights(modelfile)
