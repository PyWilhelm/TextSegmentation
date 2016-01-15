import LSTM_multilayers, LSTM_bitmap, GRN_bitmap, GRN_multilayer, BiLSTM_bitmap
import sys
import pickle
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

Model = LSTM_bitmap
weight_file = 'LSTM_bitmap.model'
model_network = Model.get_model()
model_network.load_weights(weight_file)

validate_x_file = '../' + sys.argv[1]
validate_y_file = '../' + sys.argv[2]

validate_x, validate_y = Model.prepare_data(validate_x_file, validate_y_file)

predict_y = model_network.predict({'input': validate_x})['output'] >= 0.5
print(predict_y)
shape_y = predict_y.shape
predict_y = predict_y.reshape((shape_y[0] * shape_y[1], shape_y[2]))
predict_y = (predict_y == [0])[:, 0]
validate_y = validate_y.reshape((shape_y[0] * shape_y[1], shape_y[2]))
validate_y = (validate_y == [0])[:, 0]

print(predict_y.shape, validate_y.shape)
a = precision_recall_fscore_support(validate_y, predict_y, average='binary')
print(a)
