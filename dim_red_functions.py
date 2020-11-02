import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle


def pca(X, k):
    """
    Args:
        X (np.array, dim: n,m,4): containing the image channel to be processed
        k (int): the number of elements to be used in the low rank approximation

    Returns:
        X_low_rank_approx (np.array, dim: n,m,4): low rank approximation of X
        U (np.array, dim: m,k,4): singular vectors used to obtain the low rank approximation
    """

    n, m, _ = X.shape
    U = np.zeros((m, k, 4))
    rgba = []
    for i in range(4):
        Xi = X[:, :, i]
        Mi = np.mean(Xi, axis=0)
        Ci = Xi - Mi
        _, _, Ui = np.linalg.svd(Ci)
        Uk = Ui.T[:, :k]
        rgba.append(Ci @ Uk @ Uk.T + Mi)
        U[:, :, i] = Uk
    X_low_rank_approx = np.dstack(rgba)
    return X_low_rank_approx, U


def svd(X, k):
    """
    Args:
        X (np.array, dim: n,m,4): containing the image channel to be processed
        k (int): the number of elements to be used in the low rank approximation

    Returns:
        X_low_rank_approx (np.array, dim: n,m,4): low rank approximation of X
    """

    n, m, _ = X.shape
    rgba = []
    for i in range(4):
        Xi = X[:, :, i]
        Ui, Si, Vi = np.linalg.svd(Xi)
        rgba.append(Ui[:, :k] @ np.diag(Si)[:k, :k] @ Vi[:k, :])
    X_low_rank_approx = np.dstack(rgba)
    return X_low_rank_approx


def nmf(X, k, iterations=10):
    """
    Args:
        X (np.array, dim: n,m,4): containing the image channel to be processed
        k (int): the number of elements to be used in the low rank approximation

    Returns:
        X_low_rank_approx (np.array, dim: n,m,4): low rank approximation of X
    """

    rgba = []
    n, m, _ = X.shape
    for i in range(4):
        Xi = X[:, :, i]
        Ui_0, Si_0, Vti_0 = np.linalg.svd(Xi, full_matrices=False)
        Wi = np.abs(Ui_0[:, :k])
        Hi = np.abs(Vti_0[:k, :])
        for j in range(iterations):
            # print(f'iteration {j+1} of {iterations} on dim {i}')
            Hi = Hi * (Wi.T @ Xi) / (Wi.T @ Wi @ Hi + 1e-15)
            Wi = Wi * (Xi @ Hi.T) / (Wi @ Hi @ Hi.T + 1e-15)
        rgba.append(Wi @ Hi)
    X_low_rank_approx = np.dstack(rgba)
    return X_low_rank_approx


def mds(X, k):
    """
    Args:
        X (np.array, dim: n,m,4): containing the image channel to be processed
        k (int): the number of elements to be used in the low rank approximation

    Returns:
        X_low_rank_approx (np.array, dim: n,m,4): low rank approximation of X
    """

    rgba = []
    for i in range(4):
        # print(f'dimension {i+1} of 4')
        Xi = X[:, :, i]
        Mi = np.mean(Xi, axis=0)
        Ci = Xi - Mi
        eigenValues, eigenVectors = np.linalg.eig(Ci @ Ci.T)
        idx = eigenValues.argsort()[::-1]
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:, idx]
        Qk = eigenVectors[:, :k]
        rgba.append(np.real(Qk @ Qk.T @ Ci + Mi))
    X_low_rank_approx = np.dstack(rgba)
    return X_low_rank_approx