from keras.models import Graph
from keras.layers.core import TimeDistributedDense, Dropout
from keras.layers.recurrent import LSTM
import pickle
import numpy as np
from keras.utils import np_utils
import sys

def get_model():
    print('build model')
    model = Graph()
    model.add_input(name='input', input_shape=(200, 54))
    model.add_node(LSTM(100, return_sequences=True, input_shape=(200, 54)),
                   name='forward', input='input')
    model.add_node(LSTM(100, return_sequences=True, input_shape=(200, 54), go_backwards=True),
                   name='backward', input='input')
    model.add_node(Dropout(0.2), name='dropout', inputs=['forward', 'backward'])
    model.add_node(TimeDistributedDense(2, activation='sigmoid'), name='sigmoid', input='dropout')
    model.add_output(name='output', input='sigmoid')
    print('compile')
    model.compile('adam', {'output': 'binary_crossentropy'})
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

    model = get_model()
    X_bitmap, all_y = prepare_data(train_x_file, train_y_file)
    print(all_y.shape)
    print('loaded')
    print("Train...")
    res = model.fit({'input': X_bitmap, 'output': all_y}, nb_epoch=2, validation_split=0.2)
    model.save_weights(modelfile)
