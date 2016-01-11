import LSTM_multilayers, LSTM_bitmap
import sys
import pickle
import numpy as np
from sklearn.metrics import precision_recall_curve

Model = LSTM_multilayers
weight_file = 'multilayer.model'
model_network = Model.get_model()
model_network.load_weights(weight_file)

validate_x_file = '../' + sys.argv[1]
validate_y_file = '../' + sys.argv[2]

validate_x, validate_y = Model.prepare_data(validate_x_file, validate_y_file)

acc = model_network.evaluate(validate_x, validate_y, show_accuracy=True)
print(acc)