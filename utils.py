"""
Most functions are taken directly or adapted from
https://github.com/gregcanal/multiuser-metric-preference
"""

import numpy as np
import cvxpy as cp
import scipy.special as sc

def generate_binary_responses(items, S, M, V, noise_param=4):
    """
    Inputs:
    items: item representations, d-dimensional
    S: list of comparisons, (user, (item 1, item 2))
    M: underlying metric
    V: pseudo ideal points of users
    noise_param: parameter for sigmoid function
    """

    T = len(S)
    Y = np.zeros(T)

    for t in range(T):

        k = S[t][0]         # user
        v = V[:, k]         # pseudo ideal point of user

        i,j = S[t][1]       # items to compare
        xi = items[:, i]    # embedding of item xi
        xj = items[:, j]    # embedding of item xj
        
        unquant = xi.dot(M.dot(xi)) - xj.dot(M.dot(xj)) + v.dot(xi - xj)

        if noise_param is None:
            Y[t] = np.sign(unquant)
        else:
            p = sc.expit(noise_param * unquant)     # Pr(Y = 1)
            Y[t] = np.random.choice([1, -1], p=[p, 1-p])

    return Y

def loss(X, S, Y, varM, varV, loss_fun='logistic', loss_param=1):

    if loss_fun != 'logistic' or loss_param is None:
        logistic_scale = 1
    else:
        logistic_scale = loss_param

    loss_funs = {'hinge': lambda x : cp.pos(1 - x), # cp.pos(x) = max(0, x)
                'logistic': lambda x: cp.logistic(-logistic_scale * x)} # cp.logistic(x) = log(1 + exp(x))
    cp_loss_fun = loss_funs[loss_fun] 

    mT = len(S)
    loss = 0
    for mi in range(mT):
        k = S[mi][0]

        i,j = S[mi][1]
        xi = X[:, i]
        xj = X[:, j]
        
        loss += (1/mT) * cp_loss_fun(Y[mi] * (cp.quad_form(xi, varM) - cp.quad_form(xj, varM) 
                                              + varV[:, k] @ (xi - xj)))

    return loss

def projPSD(M):

    Ms = (M + M.T)/2

    lams, V = np.linalg.eig(Ms)
    lams_trunc = np.maximum(lams, 0)

    return V @ np.diag(lams_trunc) @ V.T

def relative_error(A, Ahat):
    return np.linalg.norm(A - Ahat, 'fro') / np.linalg.norm(A, 'fro')

def rank_r_approximation(matrix, r):
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    approx = np.dot(U[:, :r], np.dot(np.diag(S[:r]), Vt[:r, :]))
    return approx