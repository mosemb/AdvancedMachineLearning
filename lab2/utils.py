import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.spatial.distance import cdist

#Function definitions for this lab

#Pay special attention to the train_krls and train_kls_early_stop functions.
#The first function solves regularized kernel least-squares (also known as kernel ridge regression or KRR) in closed form. Here the name of the algorithm suggests that the regularization is explicit, and indeed it is governed by the parameter lam (commonly 𝜆)

#The second solves kernel least-squares without regularization, using gradient descent. Here the amount of regularization is decided by the number of iterations t, such that 1𝑡≈𝜆. This means that a higher number of iterations is equivalent to less regularization.


def flip_labels(Y, perc):
    """
    Flips randomly selected labels of a binary classification problem with labels +1,-1

    Parameters
    ----------
    Y: array of labels
    perc: percentage of labels to be flipped

    Returns
    -------
    Y: array with flipped labels
    """
    if perc < 1 or perc > 100:
        raise ValueError("p should be a percentage value between 0 and 100.")

    if any(np.abs(Y) != 1):
        raise ValueError("The values of Ytr should be +1 or -1.")

    Y_noisy = np.copy(np.squeeze(Y))
    if Y_noisy.ndim > 1:
        raise ValueError("Please supply a label array with only one dimension")

    n = Y_noisy.size
    n_flips = int(np.floor(n * perc / 100))
    idx_to_flip = np.random.choice(n, size=n_flips, replace=False)
    Y_noisy[idx_to_flip] = -Y_noisy[idx_to_flip]

    return Y_noisy


def two_moons(npoints=None, pflip=0):
    """
    Read the two-moons dataset from file 'moons_dataset.mat'.
    Optionally subsample the dataset, and flips some of its labels
    
    Parameters
    ----------
    npoints : 
        The number of data-points to keep from the dataset.
        The whole dataset has 100 points.
    pflip :
        The percentage of labels to flip.
    """
    mat_contents = sio.loadmat('./moons_dataset.mat')
    Xtr = mat_contents['Xtr']
    Ytr = mat_contents['Ytr']
    Xts = mat_contents['Xts']
    Yts = mat_contents['Yts']
    if npoints is None:
        npoints = Xtr.shape[0]
    else:
        npoints = min([100, npoints])
    i = np.random.permutation(100)
    sel = i[0:npoints]
    Xtr = Xtr[sel, :]
    if pflip > 1:
        Ytrn = flip_labels(Ytr[sel], pflip)
        Ytsn = flip_labels(Yts, pflip)
    else:
        Ytrn = np.squeeze(Ytr[sel])
        Ytsn = np.squeeze(Yts)
    return Xtr, Ytrn, Xts, Ytsn


def kernel_matrix(x1, x2, kernel, kernel_args):
    """
    Parameters
    ----------
    x1, x2: collections of points on which to compute the Gram matrix
    kernel: can be 'linear', 'polynomial' or 'gaussian'
    kernel_args: 
        Parameter for the chosen kernel. Should be `None` for the linear kernel,
        the exponent for the polynomial kernel, or the length-scale for the gaussian
        kernel.

    Returns
    -------
    k: Gram matrix
    """
    if kernel == 'linear':
        k = np.dot(x1, np.transpose(x2))
    elif kernel == 'polynomial':
        k = np.power((1 + np.dot(x1, np.transpose(x2))), kernel_args)
    elif kernel == 'gaussian':
        gamma = -1 / (2 * kernel_args ** 2)
        k = np.exp(gamma * cdist(x1, x2, 'sqeuclidean'))
    return k


def train_krls(Xtr, Ytr, lam, kernel, kernel_args):
    """
    Parameters
    ----------
    Xtr: training input
    Ytr: training output
    lam: regularization parameter
    kernel: type of kernel ('linear', 'polynomial', 'gaussian')
    kernel_args: arguments to the kernel

    Returns
    -------
    c: model weights

    Examples
    --------
    ```
    c =  train_krls(Xtr, Ytr, lam=1e-1, kernel='gaussian', kernel_args=1)
    ```
    """
    n = Xtr.shape[0]
    K = kernel_matrix(Xtr, Xtr, kernel, kernel_args)
    c = np.linalg.solve(K + lam * n * np.identity(n), Ytr)

    return c


def linear_predict(c, Xtr, Xte, kernel, kernel_args):
    '''
    Parameters
    ----------
    c: model weights (as returned from the training functions)
    Xtr: training input
    Xte: test points
    kernel: type of kernel ('linear', 'polynomial', 'gaussian')
    kernel_args: 
        arguments to the kernel. 
        Note these should be the same as those used for training.

    Returns
    -------
    y_test : predicted function values on the test points

    Example of usage:
    ```
    c = train_krls(Xtr, Ytr, lam=0.1, kernel='gaussian', kernel_args=1)
    y =  linear_predict(c, Xtr, Xte, kernel='gaussian', kernel_args=1)
    ```
    '''
    k_test_train = kernel_matrix(Xte, Xtr, kernel, kernel_args)
    y = np.dot(k_test_train, c)
    return y


def c_err(y_true, y_pred):
    """
    Calculate the 0-1 loss (classification error) between true labels and predictions
    
    Parameters
    ----------
    y_true
        The array of true labels
    y_pred
        The array of predicted labels
    
    Returns
    -------
    classification_error
        The classification error. A value of 0 means the data were fitted perfectly, 
        while a value around 0.5 indicates random predictions.
    """
    return np.mean(np.sign(y_true.reshape(-1)) != np.sign(y_pred.reshape(-1)))


def plot_sep_func(c, Xtr, Ytr, Xte, kernel, kernel_args, axs):
    """
    The function classifies points evenly sampled in a visualization area,
    according to the classifier Regularized Least Squares

    Parameters
    ----------
    c: model weights
    Xtr: training input
    Ytr: training output
    Xte: test points
    kernel: type of kernel ('linear', 'polynomial', 'gaussian')
    kernel_args: width of the gaussian kernel, if used
    axs: the axis on which to draw

    Examples
    --------
    ```
    lam = 0.01
    kernel = 'gaussian'
    sigma = 1
    
    fig, ax = plt.subplots()
    c = train_krls(Xtr, Ytr, lam, 'gaussian', sigma)
    plot_sep_func(c, Xtr, Ytr, Xte, 'gaussian', sigma, ax)
    ```
    """
    step = 0.05

    x = np.arange(Xte[:, 0].min(), Xte[:, 0].max(), step)
    y = np.arange(Xte[:, 1].min(), Xte[:, 1].max(), step)

    xv, yv = np.meshgrid(x, y)

    xv = xv.flatten('F')
    xv = np.reshape(xv, (xv.shape[0], 1))

    yv = yv.flatten('F')
    yv = np.reshape(yv, (yv.shape[0], 1))

    xgrid = np.concatenate((xv, yv), axis=1)

    ygrid = linear_predict(c, Xtr, xgrid, kernel, kernel_args)

    colors = [-1, +1]
    cc = []
    for item in Ytr:
        cc.append(colors[(int(item)+1)//2])

    axs.scatter(Xtr[:, 0], Xtr[:, 1], c=Ytr,cmap = 'viridis', s=50, edgecolors = 'black')

    z = np.asarray(np.reshape(ygrid, (y.shape[0], x.shape[0]), 'F'))
    axs.contour(x, y, z, 1, colors = 'deeppink')
    
    
def train_kls_early_stop(Xtr, Ytr, t, kernel, kernel_args):
    """
    Parameters
    ----------
    Xtr: training input
    Ytr: training output
    t: Number of training iterations
    kernel: type of kernel ('linear', 'polynomial', 'gaussian')
    kernel_args: arguments to the kernel

    Returns
    -------
    c: model weights

    Examples
    --------
    ```
    c =  train_kls_early_stop(
        Xtr, Ytr, t=10, kernel='gaussian', kernel_args=1)
    ```
    """
    n = Xtr.shape[0]
    K = kernel_matrix(Xtr, Xtr, kernel, kernel_args)
    
    # The step size must be strictly less than the 
    # inverse of the maximum eigenvalue of K.
    K_eig = np.linalg.eigvalsh(K)
    max_eig = K_eig[-1]
    step_size = 0.99 / max_eig
    
    # Iterative solution
    c = np.zeros(n)
    for it in range(t):
        c = c - step_size * (K @ c - Ytr)

    return c