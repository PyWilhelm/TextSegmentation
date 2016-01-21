from keras.models import Graph
from keras.layers.core import TimeDistributedDense, Dropout
from keras.layers.recurrent import LSTM
import pickle
import numpy as np
from keras.utils import np_utils
import sys
from iterate import prepare_data


def get_model():
    print('build model')
    model = Graph()
    model.add_input(name='input', input_shape=(200, 54*5))
    model.add_node(LSTM(128, return_sequences=True, input_shape=(200, 54*5)),
                   name='forward', input='input')
    model.add_node(LSTM(128, return_sequences=True, input_shape=(200, 54*5), go_backwards=True),
                   name='backward', input='input')
    model.add_node(Dropout(0.2), name='dropout', inputs=['forward', 'backward'])
    model.add_node(TimeDistributedDense(2, activation='sigmoid'), name='sigmoid', input='dropout')
    model.add_output(name='output', input='sigmoid')
    print('compile')
    model.compile('adam', {'output': 'binary_crossentropy'})
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
