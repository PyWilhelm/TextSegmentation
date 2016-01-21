import numpy as np
import sys
import pickle
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Masking, TimeDistributedDense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from iterate import prepare_data


def get_model():
    model = Sequential()
    # model.add(Embedding(54, 32, input_length=200))
    model.add(LSTM(128, return_sequences=True, input_shape=(200, 270)))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributedDense(2))
    model.add(Activation('sigmoid'))
    print('compile')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  class_mode="binary")
    return model


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
            model.fit(x,y, nb_epoch=1, show_accuracy=True)
    model.save_weights(modelfile)
