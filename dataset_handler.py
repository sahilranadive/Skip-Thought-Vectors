

import numpy as np
from numpy.random import RandomState
import os.path


def load_data(encoder, name, loc='./data/', seed=1234):
   
    z = {}
    if name == 'MR':
        pos, neg = load_rt(loc=loc)
    elif name == 'SUBJ':
        pos, neg = load_subj(loc=loc)
    elif name == 'CR':
        pos, neg = load_cr(loc=loc)
    elif name == 'MPQA':
        pos, neg = load_mpqa(loc=loc)

    labels = compute_labels(pos, neg)
    text, labels = shuffle_data(pos+neg, labels, seed=seed)
    z['text'] = text
    z['labels'] = labels

    features = encoder.encode(text)

    return z, features


def load_rt(loc='./data/'):
   
    pos, neg = [], []
    with open(os.path.join(loc, 'rt-polarity.pos'), 'rb') as f:
        for line in f:
            pos.append(line.decode('latin-1').strip())
    with open(os.path.join(loc, 'rt-polarity.neg'), 'rb') as f:
        for line in f:
            neg.append(line.decode('latin-1').strip())
    return pos, neg


def compute_labels(pos, neg):

    labels = np.zeros(len(pos) + len(neg))
    labels[:len(pos)] = 1.0
    labels[len(pos):] = 0.0
    return labels


def shuffle_data(X, L, seed=1234):
   
    prng = RandomState(seed)
    inds = np.arange(len(X))
    prng.shuffle(inds)
    X = [X[i] for i in inds]
    L = L[inds]
    return (X, L)    




