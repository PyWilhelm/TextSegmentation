import gensim
import gensim.models
from gensim.models import word2vec
import pickle

print(word2vec.FAST_VERSION)

print(gensim.models)
raw_x = pickle.load(open('training_x.dat', 'rb'))
print(raw_x[0])
raw_y = pickle.load(open('training_y.dat', 'rb'))

model = gensim.models.Word2Vec(raw_x, workers=8, size=10)
print(model)

model.save('word2vec.dat')