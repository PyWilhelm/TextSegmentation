import pickle
import gensim.corpora.wikicorpus as wc
import bz2
from gensim import utils
import numpy as np
import sys
import os

MAX_CHAR_LENGTH = 200

alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'


def prepare_data(filename, destname):
    pages = wc.extract_pages(bz2.BZ2File(filename), ('0',))
    corpus = []
    x = []
    y = []
    count = 0
    for p in pages:
        text = wc.filter_wiki(p[1])
        tokens = [token.encode('utf8') for token in utils.tokenize(text, errors='ignore')
                  if len(token) <= 15 and not token.startswith('_')]
        if len(tokens) >= 50:
            length = 0
            old_i = 0
            for i, token in enumerate(tokens):
                length += len(token)
                if length > MAX_CHAR_LENGTH:
                    corpus.append(tokens[old_i: i])
                    length = len(token)
                    old_i = i
                if i == len(tokens) - 1:
                    corpus.append(tokens[i:])
    count = 0
    for sent in corpus:
        count += 1
        if count >= 100000:
            break
        sent_y = []
        sent_x = []
        for token in sent:
            if all([65 <= c <= 90 or 97 <= c <= 122 for c in token]):
                sent_y.extend([False] * (len(token) - 1) + [True])
                sent_x.extend([c - 64 if c <= 90 else c - 70 for c in token])
        sent_y.extend([False] * (MAX_CHAR_LENGTH - len(sent_x)))
        sent_x.extend([0] * (MAX_CHAR_LENGTH - len(sent_x)))
        y.append(sent_y)
        x.append(sent_x)
        if len(sent_x) != MAX_CHAR_LENGTH:
            print(len(sent_x))

    x = np.array(x)
    y = np.array(y)
    pickle.dump(x, open(os.path.abspath(destname + '_x'), 'wb'))
    pickle.dump(y, open(os.path.abspath(destname + '_y'), 'wb'))

if __name__ == '__main__':
    print(sys.argv)
    filename = sys.argv[1]
    destname = sys.argv[2]
    prepare_data(filename, destname)
    print('done')
