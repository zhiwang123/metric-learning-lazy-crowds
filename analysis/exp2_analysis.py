import numpy as np
import scipy.io as sio
import scipy.io as sio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

def plot(root, d, n, r, d_start):

    load_path = root + "aggregate.mat"
    results = sio.loadmat(load_path)['error'][:,d_start-2:,:]

    matplotlib.style.use('seaborn')
    plt.figure(figsize=(9, 8))

    x_values = np.arange(d_start, d+1)
    y_values = np.arange(5, n+1) 
    mean = np.mean(results, axis = 0)

    plt.imshow(mean.T, extent=[x_values.min()-0.5, x_values.max()+0.5, y_values.min(), y_values.max()], 
               origin='lower', cmap=sns.color_palette("Blues", as_cmap=True), aspect='auto')

    cbar = plt.colorbar() 
    cbar.set_label('Relative error',fontsize=25)
    cbar.ax.tick_params(labelsize=17)

    plt.grid(False)
    plt.xlabel('Ambient dimension', fontsize=25)
    plt.ylabel('Number of {}-d subspaces'.format(r), fontsize=25)
    plt.xticks(x_values, fontsize=17)
    plt.yticks(y_values[::5], fontsize=17)

    x = np.linspace(x_values.min()-0.5, x_values.max()+0.5, 100)
    y = x * (x + 1) / (r * (r + 1))
    plt.plot(x, y, color='red', linestyle='--', label='d(d+1)/{}'.format(r*(r+1)))
    plt.legend(fontsize=17)

    save_path = root + "plot.png"
    plt.savefig(save_path)

def aggregate_results(root, index_list, d, n):
    """
    Aggregate results from runs
    """
    results = {
        'error': np.zeros((len(index_list), d-1, n-4))
    }

    load_path = root + "data/exp2"
    save_path = root + "aggregate.mat"

    for index in runs:
        results_index = sio.loadmat(load_path + "_{}.mat".format(index), squeeze_me=False)
        results['error'][index] = results_index['error']

    sio.savemat(save_path, results)

if __name__ == '__main__':

    root = "examples/exp_results/exp2/"

    d = 10
    n = 80
    r = 1
    d_start = 3
    runs = list(range(30))

    aggregate_results(root, runs, d, n)
    plot(root, d, n, r, d_start)

