import numpy as np

def wasserstein(M, r, c, lam, epsilon=1e-5):

    n, m = M.shape

    # solution of the transport matrix by solving the 
    # optimization equation (minimize (Frobenius norm minus entropy)
    P = np.exp(- lam * M)
    P /= P.sum()
    u = np.zeros(n)

    # iterative normalization
    while np.max(np.abs(u - P.sum(1))) > epsilon:
        u = P.sum(1)
        P *= (r / u).reshape((-1, 1))
        P *= (c / P.sum(0)).reshape((1, -1))
    return P, np.sum(P * M)


if __name__ == '__main__':

    import matplotlib.pyplot as plt 

    # Evaluate distance between two moon-shaped feature distributions.
    # Transport matrix*Cost matrix tells us how much effort would be needed to transform
    # one feature distribution into another. The feature distributions may have different lengths (number of datapoints)

    from sklearn.datasets import make_moons
    from scipy.spatial import distance_matrix

    X, y = make_moons(n_samples=100, noise=0.1, shuffle=False)

    X1 = X[y==1,:]
    X2 = -X[y==0,:] # invert one moon so that both face same direction, make alignment easier

    n, m = X1.shape[0], X2.shape[0]

    r = np.ones(n) / n
    c = np.ones(m) / m

    M = distance_matrix(X1, X2)
    P, d = wasserstein(M, r, c, lam=30)

    fig, (ax1, ax2, ax) = plt.subplots(ncols=3)
    ax.scatter(X1[:,0], X1[:,1], color='blue')
    ax.scatter(X2[:,0], X2[:,1], color='orange')

    for i in range(n):
        p = np.argmax(P[i,:])
        ax.plot([X1[i,0], X2[p,0]], [X1[i,1], X2[p,1]], color='red', alpha=0.5)

    ax.set_title('Optimal matching by computing Wasserstein distances')

    ax1.imshow(M)
    ax1.set_title('Cost matrix')

    ax2.imshow(P)
    ax2.set_title('Transport matrix')
    plt.show()
