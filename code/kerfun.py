import numpy as np

def Kerfun(kernel, X, Z, p1, p2):
    """
    Compute kernel matrix between X and Z.

    Parameters:
        kernel (str): type of kernel function
        X (np.ndarray): m x p matrix
        Z (np.ndarray): n x p matrix
        p1 (float): parameter 1 for the kernel function
        p2 (float): parameter 2 for the kernel function

    Returns:
        np.ndarray: kernel matrix of size m x n
    """
    if X.shape[1] != Z.shape[1]:
        print('The second dimensions for X and Z must agree.')
        return None

    if kernel.lower() == 'linear':
        K = np.dot(X, Z.T)
    elif kernel.lower() == 'poly':
        K = (np.dot(X, Z.T) + p1) ** p2
    elif kernel.lower() == 'rbf':
        K = np.exp(-p1 * (np.tile(np.sum(X ** 2, axis=1), (Z.shape[0], 1)).T +
                          np.tile(np.sum(Z ** 2, axis=1), (X.shape[0], 1)) -
                          2 * np.dot(X, Z.T)))
    elif kernel.lower() == 'erbf':
        K = np.exp(-np.sqrt(np.tile(np.sum(X ** 2, axis=1), (Z.shape[0], 1)).T +
                            np.tile(np.sum(Z ** 2, axis=1), (X.shape[0], 1)) -
                            2 * np.dot(X, Z.T)) / (2 * p1 ** 2)) + p2
    elif kernel.lower() == 'sigmoid':
        K = np.tanh(p1 * np.dot(X, Z.T) / X.shape[1] + p2)
    else:
        K = np.dot(X, Z.T) + p1 + p2

    return K
