import numpy as np
from matplotlib import pyplot as plt


def plot(X,Y,pred_func):
    # positive and negative examples
    idxP = np.nonzero(Y[:,0]==1)[0];
    idxN = np.nonzero(Y[:,0]==0)[0];

    # determine canvas borders
    mins = np.amin(X,0); 
    mins = mins - 0.1*np.abs(mins);
    maxs = np.amax(X,0); 
    maxs = maxs + 0.1*maxs;

    ## generate dense grid
    [xs,ys] = np.meshgrid(np.linspace(mins[0,0],maxs[0,0],500), 
            np.linspace(mins[0,1], maxs[0,1], 500));


    # evaluate model on the dense grid
    Z = pred_func(np.c_[xs.flatten(1), ys.flatten(1)]);
    Z = Z.reshape(xs.shape)

    # Plot the contour and training examples
    plt.contourf(xs, ys, Z, cmap=plt.cm.Spectral)
    #plt.scatter(X[:, 0], X[:, 1], c=Y[:,0], cmap=plt.cm.Spectral)

    #pos = np.nonzero(z>=.5)[0];
    #neg = np.nonzero(z<.5)[0];
    #plt.plot(xs[pos], ys[pos], '.')
    #plt.plot(xs[neg], ys[neg], 'r.')

    plt.plot(X[idxP,0], X[idxP,1],'o', color='#8b0000')
    plt.plot(X[idxN,0], X[idxN,1],'bo')
    plt.show()
