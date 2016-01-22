import pickle
import numpy as np
from keras.utils import np_utils



def prepare_data(x_filename, y_filename, size=5000):

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