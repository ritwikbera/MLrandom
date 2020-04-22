import numpy as np 
from sklearn.utils.graph_shortest_path import graph_shortest_path

def dist(a, b):
    return np.sqrt(sum((a - b)**2))

def distance_mat(X, n_neighbors=6):

    distances = np.array([[dist(p1, p2) for p2 in X] for p1 in X])
    neighbors = np.zeros_like(distances)
    neighbors[:, 1:n_neighbors+1] = np.argsort(distances, axis=1)[:, 1:n_neighbors+1]
    
    # print(neighbors[:5,1:n_neighbors+1])

    return neighbors

def isomap(data, n_components=2, n_neighbors=6):

    data = distance_mat(data, n_neighbors)
    graph = graph_shortest_path(data, directed=False)
    graph = -0.5 * (graph ** 2)

    return mds(graph, n_components)

def center(K):
    n_samples = K.shape[0]

    meanrows = np.sum(K, axis=0)/n_samples
    meancols = (np.sum(K, axis=1)/n_samples)[:, np.newaxis]
    meanall = meanrows.sum() / n_samples

    K -= meanrows
    K -= meancols
    K += meanall
    return K

def mds(data, n_components=2):
    center(data)

    eig_val_cov, eig_vec_cov = np.linalg.eig(data)
    eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:, i]) for i in range(len(eig_val_cov))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    eig_pairs = np.array(eig_pairs)

    matrix_w = np.hstack([eig_pairs[i, 1].reshape(data.shape[1], 1) for i in range(n_components)])

    return matrix_w

if __name__=='__main__':
    from sklearn.datasets import load_digits
    X, _ = load_digits(return_X_y=True)
    X = np.random.randint(low=0,high=10,size=(100,10))
    print(X.shape)
    print(isomap(X).shape)
