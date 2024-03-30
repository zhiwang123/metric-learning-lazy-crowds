import numpy as np
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

def plot_error(root, K_vec, noise_levels):

    matplotlib.style.use('seaborn')
    sns.set_style("whitegrid")
    colors = sns.color_palette("muted")
    plt.figure(figsize=(9, 8))

    for i in range(0,4):
        load_path = root + "noise_{}/aggregate.mat".format(i)
        errors = sio.loadmat(load_path)['error']
        errors = errors.squeeze()

        mean = np.mean(errors, axis=0).squeeze()
        std = np.std(errors,axis=0).squeeze()

        noise_level = noise_levels[i]
        plt.plot([x for x in K_vec], mean, marker='s', markersize=6, 
                 color=colors[i], label='noise level: {}'.format(noise_level))
        plt.fill_between([x for x in K_vec], mean + std, mean - std, color=colors[i], alpha=0.2)

    plt.xlabel('Number of users per subspace', fontsize=25)
    plt.ylabel('Relative error', fontsize=25)

    plt.xticks(K_vec)
    plt.gca().set_xticklabels(K_vec, fontsize=17)
    plt.yticks(fontsize=17)
    plt.legend(fontsize=17)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(2)

    save_path = root + "plot.png"
    plt.savefig(save_path)

def aggregate_results(root, noise_index, index_list, len_K_vec):
    """
    Aggregate results from runs
    """
    results = {
        'error': np.zeros((len(index_list), len_K_vec, 1))
    }

    load_path = root + "noise_{}/data/noise_{}".format(noise_index, noise_index)
    save_path = root + "noise_{}/aggregate.mat".format(noise_index)

    for index in runs:
        results_index = sio.loadmat(load_path + "_{}.mat".format(index), squeeze_me=False)
        results['error'][index] = results_index['error']

    sio.savemat(save_path, results)

if __name__ == '__main__':

    runs = list(range(30))
    K_vec = [1, 10, 20, 40, 60, 80]
    noise_levels = [0, 0.1, 0.2, 0.3]
    root = "examples/exp_results/exp3/"

    # For each of the 4 noise levels
    for noise_index in range(4):              
        aggregate_results(root, noise_index, runs, len(K_vec))

    plot_error(root, K_vec, noise_levels)