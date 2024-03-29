
import numpy as np
from scipy.sparse import hstack
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

import dataset_handler


def eval_nested_kfold(encoder, name, loc='./data/', k=10, seed=1234, use_nb=False):

    
    z, features = dataset_handler.load_data(encoder, name, loc=loc, seed=seed)

    scan = [2**t for t in range(-3,5,1)]
    
    npts = len(z['text'])
    kf = KFold( n_splits=k,random_state=seed)
    print(kf.get_n_splits(features))
    scores = []
    t=0
    for train, test in kf.split(features):
        print("fold number ",t)
        t=t+1
        X_train = features[train]
        y_train = z['labels'][train]
        X_test = features[test]
        y_test = z['labels'][test]

        Xraw = [z['text'][i] for i in train]
        Xraw_test = [z['text'][i] for i in test]

        scanscores = []
        for s in scan:

            innerkf = KFold(n_splits=k, random_state=seed+1)
            innerscores = []
            for innertrain, innertest in innerkf.split(X_train):
        
                X_innertrain = X_train[innertrain]
                y_innertrain = y_train[innertrain]
                X_innertest = X_train[innertest]
                y_innertest = y_train[innertest]

                Xraw_innertrain = [Xraw[i] for i in innertrain]
                Xraw_innertest = [Xraw[i] for i in innertest]

                if use_nb:
                    NBtrain, NBtest = compute_nb(Xraw_innertrain, y_innertrain, Xraw_innertest)
                    X_innertrain = hstack((X_innertrain, NBtrain))
                    X_innertest = hstack((X_innertest, NBtest))

                clf = LogisticRegression(C=s,solver='sag',max_iter=1000)
                clf.fit(X_innertrain, y_innertrain)
                acc = clf.score(X_innertest, y_innertest)
                innerscores.append(acc)
                print (s, acc)

            scanscores.append(np.mean(innerscores))

        s_ind = np.argmax(scanscores)
        s = scan[s_ind]
        print(scanscores)
        print(s)
 
        if use_nb:
            NBtrain, NBtest = compute_nb(Xraw, y_train, Xraw_test)
            X_train = hstack((X_train, NBtrain))
            X_test = hstack((X_test, NBtest))
       
        clf = LogisticRegression(C=s,solver='sag',max_iter=1000)
        clf.fit(X_train, y_train)

        acc = clf.score(X_test, y_test)
        scores.append(acc)
        print(scores)

    return scores


def compute_nb(X, y, Z):
    """
    Compute NB features
    """
    labels = [int(t) for t in y]
    ptrain = [X[i] for i in range(len(labels)) if labels[i] == 0]
    ntrain = [X[i] for i in range(len(labels)) if labels[i] == 1]
    poscounts = nbsvm.build_dict(ptrain, [1,2])
    negcounts = nbsvm.build_dict(ntrain, [1,2])
    dic, r = nbsvm.compute_ratio(poscounts, negcounts)
    trainX = nbsvm.process_text(X, dic, r, [1,2])
    devX = nbsvm.process_text(Z, dic, r, [1,2])
    return trainX, devX



