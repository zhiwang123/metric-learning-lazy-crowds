import multiprocessing as mp
import numpy as np
import scipy.io as sio
import logging
import basic_simulation
import yaml
import argparse
import os

# Standard experiment
def experiment_setup(pm, path, seed):

    save_file = path + '.mat'
    log_path = path + '.log'

    d = pm['d']             # ambient dimension
    n = pm['n']             # number of subspaces
    r = pm['r']             # subspace dimension
    K_vec = pm['K_vec']     # list of numbers of users per subspace
    m_vec = pm['m_vec']     # list of numbers of comparisons per user
    y_noise = pm['y_noise']
    loss_fun = pm['loss_fun']
    loss_param = pm['loss_param']
    approx_subspace_noise = pm['approx_subspace_noise']
    reconst_method = pm['reconst_method']

    logging.basicConfig(filename=log_path, level=logging.INFO,
            format='%(process)d - %(asctime)s - %(message)s')

    results = {'error': np.zeros((len(K_vec), len(m_vec)))}

    for i in range(len(K_vec)):

        K = K_vec[i]
        logging.info(
            'Number of users per subspace {} / {}'.format(K, K_vec[-1]))

        for j in range(len(m_vec)):

            m = m_vec[j]
            logging.info('  Number of queries per user {} / {}'.format(m, m_vec[-1]))

            error = basic_simulation.run_experiment(d, n, r, K, m, approx_subspace_noise, 
                                        y_noise, loss_fun, loss_param, reconst_method, seed)
            results['error'][i, j] = error
    
        sio.savemat(save_file, results)
        logging.info('Saved to {}'.format(save_file))

def incremental_setup(pm, path, seed):

    save_file = path + '.mat'
    log_path = path + '.log'

    d = pm['d']
    n = pm['n']
    r = pm['r']
    K = pm['K_vec'][0]
    m = pm['m_vec'][0]
    y_noise = pm['y_noise']
    loss_fun = pm['loss_fun']
    loss_param = pm['loss_param']
    approx_subspace_noise = pm['approx_subspace_noise']
    reconst_method = pm['reconst_method']

    logging.basicConfig(filename=log_path, level=logging.INFO,
            format='%(process)d - %(asctime)s - %(message)s')

    results = {'error': np.zeros((d-1, n-4))}

    for d_i in range(2,d+1):

        logging.info(
            'Ambient dimension {} / {}'.format(d_i, d))

        errors = basic_simulation.run_experiment(d_i, n, r, K, m, approx_subspace_noise, 
                                    y_noise, loss_fun, loss_param, reconst_method, seed, True)
        results['error'][d_i-2] = errors
    
        sio.savemat(save_file, results)
        logging.info('Saved to {}'.format(save_file))

def parallel_run(runs, config):

    """
    Run multiple simulations in parallel
    """

    n_cores = config['n_cores']
    pm = config['params']
    root = config['root']
    type = config['type']

    if not os.path.exists(os.path.dirname(root)):
        os.makedirs(os.path.dirname(root))

    pool = mp.Pool(n_cores)
    params = [(pm, '{}_{}'.format(root, index), np.random.randint(2**31)) for index in runs]

    if type == 'normal':
        pool.starmap(experiment_setup, params)
    elif type == 'incremental':
        pool.starmap(incremental_setup, params)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Read configuration file')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    args = parser.parse_args()

    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)
        
    runs = list(range(30))
    parallel_run(runs, config)