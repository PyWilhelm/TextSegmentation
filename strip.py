import gensim.corpora.wikicorpus as wc
import numpy as np
import pickle

# Class: 0: B, 1: M, 2: E, 3: S


def generate_corpus(filename):
    wiki = wc.WikiCorpus(filename, dictionary={})
    X, Y = [], []
    for raw_list in wiki.get_texts():
        char_list = []
        label_list = []
        for word in raw_list:
            if len(word) == 1:
                label_list.append(3)
            else:
                label_list.append(0)
                for i in range(len(word) - 2):
                    label_list.append(1)
                label_list.append(2)
            char_list.extend([chr(w) for w in word])
        X.append(char_list)
        Y.append(label_list)
    return X, Y


if __name__ == '__main__':
    raw_x, raw_y = generate_corpus('/Users/wilhelm/TextSegment/enwiki-latest-pages-articles1.xml-p000000010p000010000.bz2')
    all_x = []
    all_y = []
    m, n = 3, 3
    max = 5000000
    count = 0
    for i in range(len(raw_x)):
        for j in range(len(raw_x[i])):
            all_y.append(raw_y[i][j])
            start = j - m if j - m >= 0 else 0
            end = j + n + 1 if j + n + 1 <= len(raw_x[i]) else len(raw_x[i])
            all_x.append(raw_x[i][start: end])
            count += 1
            if count == max:
                print(len(all_x), len(all_y))
                pickle.dump(raw_x, open('training_x.dat', 'wb'))
                pickle.dump(raw_y, open('training_y.dat', 'wb'))
                pickle.dump(all_x, open('training_allx2.dat', 'wb'))
                pickle.dump(all_y, open('training_ally2.dat', 'wb'))
                break
        if count == max:
            break
