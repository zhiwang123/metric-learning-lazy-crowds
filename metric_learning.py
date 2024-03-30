# Reference: https://github.com/gregcanal/multiuser-metric-preference

import utils
import cvxpy as cp
import numpy as np
from numpy.linalg import pinv
from sklearn.linear_model import HuberRegressor

cvxkwargs_global = {'solver': cp.SCS, 'eps':1e-4, 'max_iters': 100000, 'verbose': False}

def learn_subspace_metric(r, K, items_V, S, Y, hyperparams, loss_fun='logistic', loss_param=1):
    """
    Recover each subspace metric by solving a convex optimization problem
    given in the paper 
    "One for All: Simultaneous Metric and Preference Learning over Multiple Users"

    Inputs:
    r: dimension of subspace
    K: number of users
    items_V: canonical representations of items in the subspace
    S: queries
    Y: binary responses
    hyperparams: hyperparamters for the optimization problem
    loss_fun: loss_func
    loss_param: parameter if loss_fun is logistic
    """
    
    # Variables
    if r == 1:
        varM = cp.Variable((1, 1), nonneg=True) # nonnegative if 1-dim
    else:
        varM = cp.Variable((r, r), PSD=True)    # symmetric and PSD

    varV = cp.Variable((r, K))                  # pseudo ideal points

    # Loss function
    loss = utils.loss(items_V, S, Y, varM, varV, loss_fun, loss_param)

    # Constraints
    constraints = [cp.norm(varM, 'fro') <= hyperparams[0]]
    for k in range(K):
        constraints.append(cp.norm(varV[:, k], 2) <= hyperparams[1])

    prob = cp.Problem(cp.Minimize(loss), constraints)
    prob.solve(**cvxkwargs_global)

    Mhat = varM.value
    Mhat = utils.projPSD(Mhat)

    return Mhat, prob.status

def reconstruction(d, n, B_all, Qhat_all, status_all, reconstruct_method='huber'):
    """
    Stitch together subspace metrics
    
    Inputs:
    d: ambient dimension
    n: number of subspaces
    B_all: canonical orthonormal bases of the subspaces
    Qhat_all: estimators of subspace metrics
    status_all: status of numerical solver
    method: {ordinary least squares}
    """

    initialized = False

    for i in range(n):

        if status_all[i] != 'optimal' and initialized == True:
            continue
    
        if status_all[i] != 'optimal' and initialized == False and i < n-1:
            continue

        B = B_all[i]            # orthonormal basis of subspace
        Qhat = Qhat_all[i]      # estimator of subspace metric from stage 1

        vec_Qhat = Qhat.flatten(order='F')
        Pi = np.kron(B.T, B.T)

        if not initialized:
            vec_Qhat_all = vec_Qhat
            Pi_all = Pi
            initialized = True
        else:
            vec_Qhat_all = np.hstack((vec_Qhat_all, vec_Qhat))
            Pi_all = np.vstack((Pi_all, Pi))

    if reconstruct_method == 'ols':       # Ordinary least squares
        vec_Mhat = pinv(Pi_all) @ vec_Qhat_all
        Mhat = vec_Mhat.reshape(d,d)

    elif reconstruct_method == 'huber':   # Huber regression
        huber = HuberRegressor(epsilon=1.35, max_iter=10000, 
                               fit_intercept=False).fit(Pi_all, vec_Qhat_all)
        Mhat = huber.coef_.reshape(d,d)

    # Projection to PSD cone
    Mhat = utils.projPSD(Mhat)

    return Mhat