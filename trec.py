
import numpy as np
import os.path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.utils import shuffle


def evaluate(encoder, k=10, seed=1234, evalcv=True, evaltest=False, loc='./tasks/trec_data'):
    
    print ('Preparing data...')
    traintext, testtext = load_data(loc)
    train, train_labels = prepare_data(traintext)
    test, test_labels = prepare_data(testtext)
    train_labels = prepare_labels(train_labels)
    test_labels = prepare_labels(test_labels)
    train, train_labels = shuffle(train, train_labels, random_state=seed)

    print ('Computing training skipthoughts...')
    trainF = encoder.encode(train)
    
    if evalcv:
        print ('Running cross-validation...')
        interval = [2**t for t in range(-7,9,1)]     # coarse-grained
        C = eval_kfold(trainF, train_labels, k=k, scan=interval, seed=seed)

    if evaltest:
        if not evalcv:
            C = 128    

        print ('Computing testing skipthoughts...')
        testF = encoder.encode(test)

        print ('Evaluating...')
        clf = LogisticRegression(C=C,solver='lbfgs',max_iter=1000)   # Best parameter found from CV
        clf.fit(trainF, train_labels)
        yhat = clf.predict(testF)
        print ('Test accuracy: ' , str(clf.score(testF, test_labels)))


def load_data(loc='./data/'):
   
    train, test = [], []
    with open(os.path.join(loc, 'train_1000.label'), 'rb') as f:
        for line in f:
            train.append(line.strip())
    with open(os.path.join(loc, 'TREC_10.label'), 'rb') as f:
        for line in f:
            test.append(line.strip())
    return train, test


def prepare_data(text):
    
    labels = [t.split()[0] for t in text]
    labels = [l.decode().split(":")[0] for l in labels]
    
    X = [t.split()[1:] for t in text]
    X = [' '.encode().join(t) for t in X]
    return X, labels


def prepare_labels(labels):
    
    d = {}
    count = 0
    setlabels = set(labels)
    for w in setlabels:
        d[w] = count
        count += 1
    idxlabels = np.array([d[w] for w in labels])
    return idxlabels


def eval_kfold(features, labels, k=10, scan=[2**t for t in range(-7,9,1)], seed=1234):
  
    npts = len(features)
    kf = KFold( n_splits=k, random_state=seed)
    scores = []

    for s in scan:

        scanscores = []

        for train, test in kf.split(features):

            X_train = features[train]
            y_train = labels[train]
            X_test = features[test]
            y_test = labels[test]

            clf = LogisticRegression(C=s,solver='lbfgs',max_iter=1000)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            scanscores.append(score)
            print (s, score)

        scores.append(np.mean(scanscores))
        print (scores)

    s_ind = np.argmax(scores)
    s = scan[s_ind]
    print (s_ind, s)
    return s


