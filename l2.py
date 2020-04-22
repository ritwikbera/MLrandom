import numpy as np 


def L2dist(X,Y):
    n_1=X.shape[0]
    n_2=Y.shape[0]
    p=X.shape[1]
    ones=np.ones((p,1))
    x_sq=(X**2).dot(ones)[:,0]
    y_sq=(Y**2).dot(ones)[:,0]

	# Replace multiplication by a simple repeat to avoid creating large matrices that hog memory
    # If the repeat operation were instead done in the form of matrix multiplication, it would require
    # having large matrices that are mostly empty.
    X_rpt=np.repeat(x_sq,n_2).reshape((n_1,n_2))
    Y_rpt=np.repeat(y_sq,n_1).reshape((n_2,n_1)).T
    return X_rpt+Y_rpt-2*X.dot(Y.T)