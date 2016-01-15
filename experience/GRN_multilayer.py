import numpy as np
import keras
import pickle
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Masking, TimeDistributedDense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
import sys


def get_model():
    model = Sequential()
    # model.add(Embedding(54, 32, input_length=200))
    model.add(GRU(100, return_sequences=True, input_shape=(200, 54)))
    model.add(Dropout(0.2))
    model.add(GRU(100, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(TimeDistributedDense(2))
    model.add(Activation('sigmoid'))
    print('compile')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  class_mode="binary")
    return model


def prepare_data(x_filename, y_filename):
    X = pickle.load(open(x_filename, 'rb'))
    X_bitmap = np.zeros((X.shape[0], X.shape[1], 54), dtype=np.bool)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X_bitmap[i][j][X[i][j]] = 1
    y = pickle.load(open(y_filename, 'rb'))
    all_y = np.array([np_utils.to_categorical(sample, 2) for sample in y])
    return X_bitmap, all_y


if __name__ == '__main__':

    train_x_file = '../' + sys.argv[1]
    train_y_file = '../' + sys.argv[2]
    modelfile = sys.argv[3]
    # test_x_file = '../' + sys.argv[3]
    # test_y_file = '../' + sys.argv[4]

    X_bitmap, all_y = prepare_data(train_x_file, train_y_file)

    print(all_y.shape)
    print('loaded')
    model = get_model()
    print("Train...")
    model.fit(X_bitmap, all_y, nb_epoch=2,
              validation_split=0.2, show_accuracy=True)

    model.save_weights(modelfile)
