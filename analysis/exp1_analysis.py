import numpy as np
import scipy.io as sio
import scipy.io as sio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

def plot(root, K_vec, m_vec):

    load_path = root + "aggregate.mat"

    results = sio.loadmat(load_path)['error']
    mean = np.mean(results, axis = 0)
    std = np.std(results, axis = 0)

    matplotlib.style.use('seaborn')
    sns.set_style("whitegrid")

    colors = sns.color_palette("muted")
    plt.figure(figsize=(9, 8))

    for m in range(len(m_vec)):
        if m_vec[m] == 1:
            plt.plot([x for x in K_vec], mean[:,m], marker='s', markersize=6, 
                     label='m={} pair'.format(m_vec[m]), color=colors[m])
        else:
            plt.plot([x for x in K_vec], mean[:,m], marker='s', markersize=6, 
                     label='m={} pairs'.format(m_vec[m]), color=colors[m])
        plt.fill_between([x for x in K_vec], mean[:,m] + std[:,m], 
                         mean[:,m] - std[:,m], color=colors[m], alpha=0.2)

    plt.xlabel('Number of users per subspace', fontsize=25)
    plt.ylabel('Relative error', fontsize=25)

    plt.xticks(K_vec, fontsize=17)
    plt.gca().set_xticklabels(K_vec)
    plt.yticks(fontsize=17)
    plt.legend(fontsize=17)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(2)
    
    save_path = root + "plot.png"
    plt.savefig(save_path)

def aggregate_results(root, index_list, len_K_vec, len_m_vec):
    """
    Aggregate results from runs
    """
    results = {
        'error': np.zeros((len(index_list), len_K_vec, len_m_vec))
    }

    load_path = root + "data/exp1"
    save_path = root + "aggregate.mat"

    for index in runs:
        results_index = sio.loadmat(load_path + "_{}.mat".format(index), squeeze_me=False)
        results['error'][index] = results_index['error']

    sio.savemat(save_path, results)

if __name__ == '__main__':

    root = "examples/exp_results/exp1/"

    K_vec = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80]
    m_vec = [1, 2, 4, 6, 8]
    runs = list(range(30))

    aggregate_results(root, runs, len(K_vec), len(m_vec))
    plot(root, K_vec, m_vec)