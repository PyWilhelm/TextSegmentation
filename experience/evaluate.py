import LSTM_multilayers, LSTM_bitmap, GRN_bitmap, GRN_multilayer, BiLSTM_bitmap
import sys
import pickle
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import sys
from iterate import prepare_data

validate_x_file = '../' + sys.argv[1]
validate_y_file = '../' + sys.argv[2]

model = sys.argv[3]

if model != 'bilstmb':
    if model == 'lstmb1':
        Model = LSTM_bitmap
        weight_file = 'LSTM_bitmap5.model'

    elif model == 'lstmbm':
        Model = LSTM_multilayers
        weight_file = 'LSTM_multi5.model'

    elif model == 'grnb1':
        Model = GRN_bitmap
        weight_file = 'GRN_bitmap5.model'

    elif model == 'grnbm':
        Model = GRN_multilayer
        weight_file = 'GRN_multi5.model'

    model_network = Model.get_model()
    model_network.load_weights(weight_file)

    iterator = prepare_data(validate_x_file, validate_y_file)

    results = []
    for validate_x, validate_y in iterator:
        predict_y = model_network.predict_classes(validate_x)
        shape_y = predict_y.shape
        predict_y = predict_y.reshape((shape_y[0] * shape_y[1], shape_y[2]))
        predict_y = (predict_y == [0])[:, 0]
        validate_y = validate_y.reshape((shape_y[0] * shape_y[1], shape_y[2]))
        validate_y = (validate_y == [0])[:, 0]
        print(predict_y.shape, validate_y.shape)
        a = precision_recall_fscore_support(validate_y, predict_y, average='binary')
        print(a)
        results.append(a[:3])

    results = np.array(results)
    print('final result:')
    print(np.average(results, axis=0))

else:
    Model = BiLSTM_bitmap
    weight_file = 'BiLSTM_bitmap5.model'

    model_network = Model.get_model()
    model_network.load_weights(weight_file)

    iterator = prepare_data(validate_x_file, validate_y_file)

