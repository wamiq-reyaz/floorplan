import matplotlib.pyplot as plt
plt.style.use('ggplot')

from scipy.stats import chisquare, wasserstein_distance

import pickle
from easydict import EasyDict as ED
from node    import Floor

if __name__ == '__main__':
        floor = Floor()
        data_stats_file = './data_stats.pkl'
        sample_stats_file = './2020_09_13__19_01_383000_samples_temp1_not_rot_39.pkl'

        with open(sample_stats_file, 'rb') as fd:
            sample_stats = pickle.load(fd)

        with open(data_stats_file, 'rb') as fd:
            data_stats = pickle.load(fd)

        for ii in range(len(floor._names)):
            print(ii)
            plt.figure(dpi=300)
            hist_real, _, patches1 = plt.hist(data_stats.heights[ii], bins=20, range=(0, 64), lw=0, alpha=0.8, histtype='stepfilled',
                                      label=' Real {}'.format(floor._names[ii]), density=True)
            plt.hist(data_stats.heights[ii], bins=20, range=(0, 64), lw=1, alpha=1.0, histtype='step', ec=patches1[0].get_facecolor(), density=True)
            hist_sampled, _, patches2 = plt.hist(sample_stats.heights[ii], bins=20, range=(0, 64), lw=0, alpha=0.8, histtype='stepfilled',
                                      label='Sampled {}'.format(floor._names[ii]), density=True)
            plt.hist(sample_stats.heights[ii], bins=20, range=(0, 64), lw=1, alpha=1.0, histtype='step', ec=patches2[0].get_facecolor(), density=True)

            emd = wasserstein_distance(hist_real, hist_sampled)
            # chi_dist = chisquare(hist_sampled, hist_real)
            # plt.text(0.5, 0.5, f'xi2={chi_dist[0]:0.3f}', transform=plt.gca().transAxes)
            plt.text(0.5, 0.6, f'emd={emd:0.4f}', transform=plt.gca().transAxes)
            plt.legend()

            plt.savefig(f'heights_ranged_{ii}.png', dpi=300)