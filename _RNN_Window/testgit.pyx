import gensim
import gensim.models
from gensim.models import word2vec
import pickle
import numpy as np


print(word2vec.FAST_VERSION)

raw_x = pickle.load(open('../char_x', 'rb'))
raw_x = np.array(raw_x)
print(raw_x.shape)
# raw_x = raw_x.flatten()
# print(raw_x.shape)
print('loaded')
model = gensim.models.Word2Vec(raw_x, workers=8, size=10)
print(model)

model.save('word2vec.dat')