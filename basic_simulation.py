import numpy as np
import data_generation, metric_learning, utils

def run_experiment(d, n, r, K, m, approx_subspace_noise=0, y_noise=4, loss_fun='logistic', 
                   loss_param=1, reconstruct_method='huber', seed=None, incremental=False):

    """
    Inputs:

    # d: ambient dimension
    # n: number of subspaces
    # r: dimension of each subspace, r < n
    # K: number of users per subspace
    # m: number of pairwise comparisons per user

    # approx_subspace_noise: if nonzero, items do not lie exactly in each subspace
    # y_noise: noise param for generating labels
    # loss_fun: used for learning subspace metrics (logistic or hinge)
    # loss_param: used in loss_fun if logistic
    # reconstruct_method: used for reconstructing M from subspace metrics, huber or ols
    # seed: random seed

    # incremental: if True, return an array of recovery errors obtained by 
                   reconstructing from an increasing number of subspaces
    """

    np.random.seed(seed)
    
    # generate metric with matrix representation M
    M = data_generation.generate_metric(d)
    
    # generate n r-dimensional subspaces
    B_all = data_generation.generate_subspaces(d, n, r)

    Qhat_all = []           # estimates of subspace metrics
    status_all = []         # status of solver for each subspace

    """
    Stage 1: Learn subspace metrics
    """
    for i in range(n):

        # orthonormal basis of the subspace
        B = B_all[i]

        # Generate data (user, items, comparisons, responses)
        # V: pseudo ideal points, r-by-K
        # items_V: canonical representation of the items in subspace (possibly perturbed)
        # S: list of queries, each query: (user, (item1, item2))
        # Y: list of (noisy) binary responses for S
        # B: basis of perturbed subspace (different from B if approx_subspace_noise > 0)
        V, items_V, S, Y, B_perturbed = \
            data_generation.generate_data(d, M, r, B, K, m, approx_subspace_noise, y_noise)
        
        if approx_subspace_noise > 0:
            B = B_perturbed
            B_all[i] = B
        
        # ground-truth subspace metric
        Q = B.T @ M @ B     
        # oracle hyperparameters
        hyperparams = [np.linalg.norm(Q, 'fro'),
                           max([np.linalg.norm(v) for v in (B.T @ V).T])]
        
        Qhat, status = \
            metric_learning.learn_subspace_metric(r, K, items_V, S, Y, hyperparams, 
                                                  loss_fun, loss_param)
        
        Qhat_all.append(Qhat)
        status_all.append(status)

    """
    Stage 2: Reconstruct M from subspace metrics
    """

    if incremental == False: 
        # Default, normal experiment setting
        Mhat = metric_learning.reconstruction(d, n, B_all, Qhat_all, status_all,
                                            reconstruct_method)
        relative_error = utils.relative_error(M, Mhat)

        return relative_error
    
    else:
        # Compute relative errors using subsets of available subspaces
        # Reconstruct using N subspaces where N is in {5, 6, 7, ..., n} 
        assert n >= 5
        relative_errors = np.zeros(n-4)

        for n_i in range(5,n+1):
            Mhat = metric_learning.reconstruction(d, n_i, B_all, Qhat_all, status_all,
                                                  reconstruct_method)
            relative_error = utils.relative_error(M, Mhat)
            relative_errors[n_i-5] = relative_error

        return relative_errors


if __name__ == '__main__':

    error = run_experiment(d=10, n=80, r=1, K=60, m=4)
    print(error)