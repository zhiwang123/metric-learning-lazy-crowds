# Reference: https://github.com/gregcanal/multiuser-metric-preference

import utils
import numpy as np
from numpy.linalg import matrix_rank
from numpy.random import multivariate_normal

def generate_metric(d):

    # Generate a positive definite matrix with Frobenius norm d
    L = np.random.multivariate_normal(np.zeros(d), np.eye(d), d)
    M = L @ L.T
    M = M * (d / np.linalg.norm(M, 'fro'))

    return M

def generate_subspaces(d, n, r):
    
    # Randomly generate n r-dimensional subspaces
    # Each subspace will have a canonical orthonomal basis
    B = []

    for _ in range(n):

        while True:
            random_vectors = np.random.multivariate_normal(np.zeros(d), 1/d * np.eye(d), r).T
            if matrix_rank(random_vectors) == r:
                break

        # Use QR decomposition to obtain an orthonormal basis
        subspace_basis, _ = np.linalg.qr(random_vectors) 
        B.append(subspace_basis)

    return np.array(B)

def generate_data(d, M, r, B, K, m, approx_subspace_noise=0, y_noise=4):
    """ 
    Inputs:
    # d: ambient dimension
    # M: matrix representation of ground-truth metric
    # r: dimension of subspace
    # B: orthonormal basis of subspace 
    # K: number of users
    # m: number of pairwise comparisons per user
    # approx_subspace_noise: items do not lie exactly in each subspace
    # y_noise: noise parameter for generating binary responses
    """
    
    # User ideal points and pseudo ideal points
    U = np.random.multivariate_normal(np.zeros(d), 1/d * np.eye(d), K).T
    V = -2 * M @ U

    # Number of items to generate
    N = 2 * K * m   

    # Generate item embeddings
    random_vectors = multivariate_normal(np.zeros(d), 1/r * np.eye(d), N).T
    items = B @ B.T @ random_vectors
    # Canonical representations of the items in the subspace
    items_V = B.T @ items

    if approx_subspace_noise > 0:
        # Perturb the items with noise
        eps = approx_subspace_noise
        items += (np.eye(d) - B @ B.T) @ multivariate_normal(np.zeros(d), 
                                                             eps * eps /(d-r) * np.eye(d), N).T

        # Find rank-r approximations of the perturbed items using SVD
        rank_r_items = utils.rank_r_approximation(items, r)
        
        # Update basis
        B_perturbed, _ = np.linalg.qr(rank_r_items)
        B = B_perturbed[:,:r]
        # Canonical representations of the perturbed items in the perturbed subspace
        items_V = B.T @ items

    S = []
    for user in range(K):
        for i in range(m):
            S.append((user, (user * 2 * m + 2 * i, user * 2 * m + 2 * i + 1)))

    Y = utils.generate_binary_responses(items, S, M, V, y_noise)

    return V, items_V, S, Y, B