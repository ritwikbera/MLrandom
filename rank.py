import numpy as np 
from sklearn.svm import SVC
from itertools import combinations
from functools import total_ordering 

def bubbleSort(arr): 
    n = len(arr) 

    for i in range(n-1): 
        for j in range(0, n-i-1): 
            if arr[j] > arr[j+1] : 
                arr[j], arr[j+1] = arr[j+1], arr[j] 

@total_ordering
class item:
    def __init__(self, featvec, clf):
        self.featvec = featvec
        self.clf = clf

    def __gt__(self, other):
        return True if self.clf.predict(np.concatenate((self.featvec, other.featvec)).reshape(1,-1)) == 1 else False

def transform_data(X, true_ranks):
    X_ = np.zeros(((X.shape[0]*(X.shape[0]-1))//2, X.shape[-1]*2))
    y = np.zeros(X_.shape[0])
    for k, (i,j) in enumerate(combinations(range(X.shape[0]), 2)):
        X_[k] = np.concatenate((X[i], X[j]))
        y[k] = int(not(true_ranks[i] < true_ranks[j]))+1
    return X_, y


if __name__=='__main__':
    np.random.seed(1)
    n_samples, n_features = 20, 5
    X = np.random.randn(n_samples, n_features)
    true_ranks = np.random.permutation(n_samples) # avoids having just one class
    
    print(X)
    print(true_ranks)

    clf = SVC(gamma='auto')
    X_, y = transform_data(X, true_ranks)
    print(X_.shape)
    print(y.shape)
    clf.fit(X_,y)

    arr = [item(X[i], clf) for i in range(n_samples)]
    bubbleSort(arr)
    [print(arr[i].featvec) for i in range(len(arr))]

