import gensim
import gensim.models
from gensim.models import word2vec
import pickle

print(word2vec.FAST_VERSION)

raw_x = pickle.load(open('training_x.dat', 'rb'))
print('loaded')

model = gensim.models.Word2Vec(raw_x, workers=8, size=10)
print(model)

model.save('word2vec.dat')