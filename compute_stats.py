import os
import math
import numpy as np
import scipy.sparse.csgraph as csg
import scipy.stats as sps
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'figure.max_open_warning': 0}) # to avoid warnings that too many plots are open (apparently this also counts number of open axes)
# from dataset import load_label_colors, load_egraph
# from options import import_legacy_train_options

from load_boxes import get_sample_names, load_boxes
from convert_boxes_to_rooms import convert_boxes_to_rooms

# compute egraph statistics
def egraph_stats(points, point_type, adj, label_count, exclude_types=None):

    stats = {}

    if (points is not None and points.dim() != 3) or \
       (point_type is not None and point_type.dim() != 3) or \
       (adj is not None and adj.dim() != 3):
        raise ValueError('each egraph property must be a batched tensor.')

    if point_type is None:
        raise ValueError('Need point type.')

    points = points.clone().cpu().numpy() if points is not None else None
    point_type = point_type.clone().cpu().numpy() if point_type is not None else None
    adj = adj.clone().cpu().numpy() if adj is not None else None

    # one-hot to index
    if point_type is not None:
        if point_type.shape[1] != 1:
            point_type = point_type.argmax(axis=1).reshape([point_type.shape[0], 1, -1])

    # symmetrize and binarize adjacency
    if adj is not None:
        adj = adj + adj.swapaxes(1, 2) >= 1

    batch_size = points.shape[0] if points is not None else (point_type.shape[0] if point_type is not None else adj.shape[0])

    if point_type is not None:
        stats['type_count'] = []

    if points is not None:
        stats['center_x'] = []
        stats['center_y'] = []
        stats['area'] = []
        stats['aspect'] = []
        stats['center_x_dist'] = []
        stats['center_y_dist'] = []
        stats['gap'] = []
        stats['center_align_best'] = []
        stats['center_align_worst'] = []
        stats['side_align_best'] = []
        stats['side_align_second_best'] = []
        stats['side_align_second_worst'] = []
        stats['side_align_worst'] = []

    if adj is not None:
        stats['neighbor_count_hist'] = []
        stats['neighbor_type_hist'] = []
        stats['unreachable'] = []
        stats['exterior_dist'] = []
        stats['type_dist'] = []

    if points is not None and adj is not None:
        stats['neighbor_center_x_dist'] = []
        stats['neighbor_center_y_dist'] = []
        stats['neighbor_gap'] = []
        stats['neighbor_center_align_best'] = []
        stats['neighbor_center_align_worst'] = []
        stats['neighbor_side_align_best'] = []
        stats['neighbor_side_align_second_best'] = []
        stats['neighbor_side_align_second_worst'] = []
        stats['neighbor_side_align_worst'] = []

    for b in range(batch_size):
        pt = point_type[b, 0, :] if point_type is not None else None
        p = points[b, :, :] if points is not None else None
        a = adj[b, :, :] if adj is not None else None

        # # undo whitening and feature scaling (afterwards 1 unit should be 1 meter)
        # if p is not None:
        #     active_mask = pt > 0.5
        #     if node_mean is not None or node_std is not None:
        #         p[:, active_mask] = (p[:, active_mask] * node_std.numpy()[0, :, np.newaxis] + node_mean.numpy()[0, :, np.newaxis])
        #     if node_feature_scaling is not None:
        #         p[:, active_mask] = p[:, active_mask] / node_feature_scaling

        # check for empty floor plans
        if p.size == 0:
            print('** Warning: skipping a floor plan without rooms. **')
            continue

        if exclude_types is not None:
            include_mask = np.all([pt != et for et in exclude_types], axis=0)
            pt = pt[include_mask] if pt is not None else None
            p = p[:, include_mask] if p is not None else None
            a = a[:, include_mask][include_mask, :] if a is not None else None

        exterior_mask = pt == 1

        if (pt == 1).all():
            print('** Warning: skipping a floor plan with only exterior. **')
            continue

        if p is not None:
            x = p[0:1, :]
            y = p[1:2, :]
            w = p[2:3, :]
            h = p[3:4, :]

            left = x - w/2
            right = x + w/2
            bottom = y - h/2
            top = y + h/2

        if pt is not None:
            type_bin_centers, type_bin_edges = hist_bins_uniform(0, label_count-1, label_count)

            stats['type_count'].append({
                'x': type_bin_centers,
                'y': np.histogram(pt, bins=type_bin_edges)[0]})

        if p is not None:

            pair_mask = np.ones([p.shape[1], p.shape[1]], dtype=bool)
            pair_mask[np.tril_indices(pair_mask.shape[0], -1)] = False
            # exclude exterior from pairs
            pair_mask[:, exterior_mask] = False
            pair_mask[exterior_mask, :] = False
            num_pairs = pair_mask.sum()

            val_bin_centers, val_bin_edges = hist_bins_uniform(-25, 25, 21, negative_overflow=True, positive_overflow=True)

            stats['center_x'].append({
                'x': (type_bin_centers, val_bin_centers),
                'y': np.histogram2d(pt, x[0], bins=[type_bin_edges, val_bin_edges])[0]})
            stats['center_y'].append({
                'x': (type_bin_centers, val_bin_centers),
                'y': np.histogram2d(pt, y[0], bins=[type_bin_edges, val_bin_edges])[0]})

            val_bin_centers, val_bin_edges = hist_bins_uniform(0, 60, 21, negative_overflow=True, positive_overflow=True)
            stats['area'].append({
                'x': (type_bin_centers, val_bin_centers),
                'y': np.histogram2d(pt, w[0] * h[0], bins=[type_bin_edges, val_bin_edges])[0]})

            val_bin_centers, val_bin_edges = hist_bins_uniform(0, 1, 21, negative_overflow=True, positive_overflow=True)
            max_len = p[2:4, :].max(axis=0)
            min_len = p[2:4, :].min(axis=0)
            aspect = np.zeros(max_len.shape)
            mask = max_len > 0
            aspect[mask] = min_len[mask] / max_len[mask]
            stats['aspect'].append({
                'x': (type_bin_centers, val_bin_centers),
                'y': np.histogram2d(pt, aspect, bins=[type_bin_edges, val_bin_edges])[0]})

            # axis-aligned center distance
            val_bin_centers, val_bin_edges = hist_bins_uniform(0, 40, 41, negative_overflow=False, positive_overflow=True)
            offsets = x - x.transpose()
            offsets = np.absolute(offsets[pair_mask])
            stats['center_x_dist'].append({
                'x': val_bin_centers,
                'y': np.histogram(offsets, bins=val_bin_edges)[0] / num_pairs})
            offsets = y - y.transpose()
            offsets = np.absolute(offsets[pair_mask])
            stats['center_y_dist'].append({
                'x': val_bin_centers,
                'y': np.histogram(offsets, bins=val_bin_edges)[0] / num_pairs})

            # axis-aligned gap
            # (this gap definition contiues shrinking as a smaller object moves into a larger one)
            val_bin_centers, val_bin_edges = hist_bins_uniform(-20, 20, 21, negative_overflow=True, positive_overflow=True)
            gap_x = np.maximum(left - right.transpose(),
                               left.transpose() - right)
            gap_y = np.maximum(bottom - top.transpose(),
                               bottom.transpose() - top)
            gap = np.maximum(gap_x, gap_y)
            gap = gap[pair_mask]
            stats['gap'].append({
                'x': val_bin_centers,
                'y': np.histogram(gap, bins=val_bin_edges)[0] / num_pairs})

            # alignment
            val_bin_centers, val_bin_edges = hist_bins_uniform(0, 20, 41, negative_overflow=False, positive_overflow=True)
            dist_center_x = np.absolute((x - x.transpose())[pair_mask])
            dist_center_y = np.absolute((y - y.transpose())[pair_mask])
            dist_center = np.stack([dist_center_x, dist_center_y])
            stats['center_align_best'].append({
                'x': val_bin_centers,
                'y': np.histogram(dist_center.min(axis=0), bins=val_bin_edges)[0] / num_pairs})
            stats['center_align_worst'].append({
                'x': val_bin_centers,
                'y': np.histogram(dist_center.max(axis=0), bins=val_bin_edges)[0] / num_pairs})

            dist_side_left = np.absolute((left - left.transpose())[pair_mask])
            dist_side_right = np.absolute((right - right.transpose())[pair_mask])
            dist_side_bottom = np.absolute((bottom - bottom.transpose())[pair_mask])
            dist_side_top = np.absolute((top - top.transpose())[pair_mask])
            dist_side = np.stack([dist_side_left, dist_side_right, dist_side_bottom, dist_side_top])
            dist_side.sort(axis=0)
            stats['side_align_best'].append({
                'x': val_bin_centers,
                'y': np.histogram(dist_side[0, :], bins=val_bin_edges)[0] / num_pairs})
            stats['side_align_second_best'].append({
                'x': val_bin_centers,
                'y': np.histogram(dist_side[1, :], bins=val_bin_edges)[0] / num_pairs})
            stats['side_align_second_worst'].append({
                'x': val_bin_centers,
                'y': np.histogram(dist_side[2, :], bins=val_bin_edges)[0] / num_pairs})
            stats['side_align_worst'].append({
                'x': val_bin_centers,
                'y': np.histogram(dist_side[3, :], bins=val_bin_edges)[0] / num_pairs})

        if a is not None:

            val_bin_centers, val_bin_edges = hist_bins_uniform(0, 20, 21, negative_overflow=False, positive_overflow=True)
            stats['neighbor_count_hist'].append({
                'x': (type_bin_centers, val_bin_centers),
                'y': np.histogram2d(pt, a.sum(axis=0), bins=[type_bin_edges, val_bin_edges])[0]})

            idx1, idx2 = np.nonzero(a)
            stats['neighbor_type_hist'].append({
                'x': (type_bin_centers, type_bin_centers),
                'y': np.histogram2d(pt[idx1], pt[idx2], bins=[type_bin_edges]*2)[0]})

            val_bin_centers, val_bin_edges = hist_bins_uniform(0, 10, 11, negative_overflow=False, positive_overflow=True)
            graph = csg.csgraph_from_dense(a)
            gdists = csg.shortest_path(graph, directed=False, unweighted=True)
            exterior_inds, = np.nonzero(pt == 1)
            if exterior_inds.size > 0:
                exterior_dists = np.min(gdists[exterior_inds, :], axis=0)
            else:
                exterior_dists = np.full(shape=[gdists.shape[1]], fill_value=np.inf)
            unreachable_mask = exterior_dists == np.inf
            stats['unreachable'].append({
                'x': type_bin_centers,
                'y': np.histogram(pt[unreachable_mask], bins=type_bin_edges)[0]})
            reachable_mask = np.logical_not(unreachable_mask)
            stats['exterior_dist'].append({
                'x': (type_bin_centers, val_bin_centers),
                'y': np.histogram2d(pt[reachable_mask], exterior_dists[reachable_mask], bins=[type_bin_edges, val_bin_edges])[0]})

            reachable_mask = gdists != np.inf
            idx1, idx2 = np.nonzero(reachable_mask)
            stats['type_dist'].append({
                'x': (type_bin_centers, type_bin_centers, val_bin_centers),
                'y': np.histogramdd([pt[idx1], pt[idx2], gdists[reachable_mask]], bins=[type_bin_edges, type_bin_edges, val_bin_edges])[0]})

        if p is not None and a is not None:

            pair_mask = a.copy()
            pair_mask[np.tril_indices(pair_mask.shape[0], -1)] = False
            # exclude exterior from pairs
            pair_mask[:, exterior_mask] = False
            pair_mask[exterior_mask, :] = False
            num_pairs = pair_mask.sum()

            # axis-aligned center distance
            val_bin_centers, val_bin_edges = hist_bins_uniform(0, 40, 41, negative_overflow=False, positive_overflow=True)
            offsets = x - x.transpose()
            offsets = np.absolute(offsets[pair_mask])
            stats['neighbor_center_x_dist'].append({
                'x': val_bin_centers,
                'y': np.histogram(offsets, bins=val_bin_edges)[0] / (num_pairs if num_pairs > 0 else 1)})
            offsets = y - y.transpose()
            offsets = np.absolute(offsets[pair_mask])
            stats['neighbor_center_y_dist'].append({
                'x': val_bin_centers,
                'y': np.histogram(offsets, bins=val_bin_edges)[0] / (num_pairs if num_pairs > 0 else 1)})

            # axis-aligned gap
            # (this gap definition contiues shrinking as a smaller object moves into a larger one)
            val_bin_centers, val_bin_edges = hist_bins_uniform(-20, 20, 21, negative_overflow=True, positive_overflow=True)
            gap_x = np.maximum(left - right.transpose(),
                               left.transpose() - right)
            gap_y = np.maximum(bottom - top.transpose(),
                               bottom.transpose() - top)
            gap = np.maximum(gap_x, gap_y)
            gap = gap[pair_mask]
            stats['neighbor_gap'].append({
                'x': val_bin_centers,
                'y': np.histogram(gap, bins=val_bin_edges)[0] / (num_pairs if num_pairs > 0 else 1)})

            # alignment
            val_bin_centers, val_bin_edges = hist_bins_uniform(0, 20, 41, negative_overflow=False, positive_overflow=True)
            dist_center_x = np.absolute((x - x.transpose())[pair_mask])
            dist_center_y = np.absolute((y - y.transpose())[pair_mask])
            dist_center = np.stack([dist_center_x, dist_center_y])
            stats['neighbor_center_align_best'].append({
                'x': val_bin_centers,
                'y': np.histogram(dist_center.min(axis=0), bins=val_bin_edges)[0] / (num_pairs if num_pairs > 0 else 1)})
            stats['neighbor_center_align_worst'].append({
                'x': val_bin_centers,
                'y': np.histogram(dist_center.max(axis=0), bins=val_bin_edges)[0] / (num_pairs if num_pairs > 0 else 1)})

            dist_side_left = np.absolute((left - left.transpose())[pair_mask])
            dist_side_right = np.absolute((right - right.transpose())[pair_mask])
            dist_side_bottom = np.absolute((bottom - bottom.transpose())[pair_mask])
            dist_side_top = np.absolute((top - top.transpose())[pair_mask])
            dist_side = np.stack([dist_side_left, dist_side_right, dist_side_bottom, dist_side_top])
            dist_side.sort(axis=0)
            stats['neighbor_side_align_best'].append({
                'x': val_bin_centers,
                'y': np.histogram(dist_side[0, :], bins=val_bin_edges)[0] / (num_pairs if num_pairs > 0 else 1)})
            stats['neighbor_side_align_second_best'].append({
                'x': val_bin_centers,
                'y': np.histogram(dist_side[1, :], bins=val_bin_edges)[0] / (num_pairs if num_pairs > 0 else 1)})
            stats['neighbor_side_align_second_worst'].append({
                'x': val_bin_centers,
                'y': np.histogram(dist_side[2, :], bins=val_bin_edges)[0] / (num_pairs if num_pairs > 0 else 1)})
            stats['neighbor_side_align_worst'].append({
                'x': val_bin_centers,
                'y': np.histogram(dist_side[3, :], bins=val_bin_edges)[0] / (num_pairs if num_pairs > 0 else 1)})

    return stats

def egraph_set_stats(stats, label_count):

    set_stats = {}
    for stat_name, stat in stats.items():

        # stack into single ndarray
        stat_x = stat[0]['x'] # all should have the same x
        stat = np.stack([s['y'] for s in stat])

        if stat_name == 'type_count':
            type_bin_centers, type_bin_edges = hist_bins_uniform(0, label_count-1, label_count)
            val_bin_centers, val_bin_edges = hist_bins_uniform(0, 40, 41, negative_overflow=False, positive_overflow=True)
            _, type_idx = np.indices(stat.shape)
            # floorplan statistics
            # each histogram is (n_types)
            type_idx = stat_x[type_idx]
            set_stats['type_freq'] = {'x': stat_x, 'y': stat.mean(axis=0)}
            set_stats['type_hist'] = {
                'x': (type_bin_centers, val_bin_centers),
                'y': np.histogram2d(type_idx.reshape(-1), stat.reshape(-1), bins=[type_bin_edges, val_bin_edges])[0] / stat.shape[0]}
        elif stat_name == 'unreachable':
            # floorplan statistics
            # each histogram is (n_types)
            set_stats[stat_name] = {'x': stat_x, 'y': stat.mean(axis=0)}
        elif stat_name in ['center_x', 'center_y', 'area', 'aspect', 'neighbor_count_hist', 'neighbor_type_hist', 'exterior_dist']:
            # single room statistics
            # each histogram is (n_types x n_value_bins)
            # weight: mass distribution of the histogram among room types (that is factored out with normalize_histograms)

            # neighbor_type_hist is the distribution of neighboring room types for each starting room type.
            # This is normalized to sum up to one per starting room type here (each row in the vis. sums to 1)

            stat_mean = stat.mean(axis=0) # mean over all egraphs
            norm_fac = stat_mean.sum(axis=1) # normalize by total count per node type (sum over counts in all bins for each type)
            mask = norm_fac > 0
            stat_mean_norm = np.zeros_like(stat_mean)
            stat_mean_norm[mask, :] = stat_mean[mask, :] / np.expand_dims(norm_fac, -1)[mask, :]

            set_stats[stat_name] = {
                'x': stat_x,
                'y': stat_mean_norm,
                'weight': norm_fac}
        elif stat_name in [
                'center_x_dist', 'center_y_dist', 'gap', 'center_align_best', 'center_align_worst',
                'side_align_best', 'side_align_second_best', 'side_align_second_worst', 'side_align_worst',
                'neighbor_center_x_dist', 'neighbor_center_y_dist', 'neighbor_gap', 'neighbor_center_align_best', 'neighbor_center_align_worst',
                'neighbor_side_align_best', 'neighbor_side_align_second_best', 'neighbor_side_align_second_worst', 'neighbor_side_align_worst']:
            # room pair statistics (have already been normalized over room pairs in each floor plan)
            # each histogram is (n_value_bins)
            set_stats[stat_name] = {'x': stat_x, 'y': stat.mean(axis=0)}
        elif stat_name == 'type_dist':
            # room pair statistics per type
            # each histogram is (n_types x n_types x n_distance_bins)
            stat_sum = stat.sum(axis=0) # sum over all egraphs
            norm_fac = stat_sum.sum(axis=2) # normalize by number of pairs for each node type
            mask = norm_fac > 0.0001
            stat_sum_dist = (stat_sum * stat_x[2]).sum(axis=2) # sum all distances in each type pair
            stat_sum_dist_norm = np.zeros_like(stat_sum_dist)
            stat_sum_dist_norm[mask] = stat_sum_dist[mask] / norm_fac[mask] # take average over all distances (from all floor plans) in each type pair

            set_stats[stat_name] = {
                'x': stat_x,
                'y': stat_sum_dist_norm,
                'weight': norm_fac}
            # set_stats[stat_name]['y'][mask] = 0 # set entries that have 0 examples from nan to zero
        else:
            raise ValueError(f'Unknown stat: {stat_name}')

    return set_stats

def hist_bins_uniform(first_center, last_center, count, negative_overflow=False, positive_overflow=False):

    bin_centers = np.linspace(first_center, last_center, count)

    bin_spacing = (last_center - first_center) / (count-1)
    bin_edges = np.linspace(first_center - bin_spacing/2, last_center + bin_spacing/2, count+1)

    if negative_overflow:
        bin_centers = np.append(first_center-bin_spacing, bin_centers)
        bin_edges = np.append(-np.inf, bin_edges)

    if positive_overflow:
        bin_centers = np.append(bin_centers, last_center+bin_spacing)
        bin_edges = np.append(bin_edges, np.inf)

    return bin_centers, bin_edges

# # normalize multiple histograms of shape nbins x nhists
# def normalize_histograms(stat):

#     hist_sum = stat.sum(axis=1, keepdims=True)
#     mask = hist_sum[:, 0] > 0
#     stat[mask, :] /= hist_sum[mask, :]

#     return stat

def egraph_set_stat_dists(stats1, stats2):

    # if 'type_freq' not in stats1:
    #     raise ValueError('Need the type_freq stat.')

    # type_freq1 = stats1['type_freq']['y']
    # type_freq2 = stats2['type_freq']['y']

    dists = {}
    for stat_name, stat1 in stats1.items():
        stat_x = stat1['x']
        if 'weight' in stat1:
            stat1_w = stat1['weight']
            stat2_w = stats2[stat_name]['weight']
        else:
            stat1_w = None
            stat2_w = None
        stat1 = stat1['y']
        stat2 = stats2[stat_name]['y']
        dists[stat_name] = 0

        if stat_name in ['type_freq', 'unreachable']:
            dists[stat_name] += ((stat1 - stat2)**2).sum()
        elif stat_name == 'type_hist':
            # max_ind = stat1.shape[1]-1
            # stat_x = np.linspace(0, max_ind, max_ind+1)
            stat_x_val = stat_x[1] / (stat_x[1].max() - stat_x[1].min()) # normalized so max. distance between x values is 1

            for c in range(stat1.shape[0]):
                if stat1[c, :].sum() <= 0 or stat2[c, :].sum() <= 0:
                    dists[stat_name] += abs(stat2[c, :].sum() - stat1[c, :].sum())
                else:
                    dists[stat_name] += sps.wasserstein_distance(
                        u_values=stat_x_val, v_values=stat_x_val, u_weights=stat1[c, :], v_weights=stat2[c, :])

            dists[stat_name] /= stat1.shape[0]
        elif stat_name in ['type_dist']:
            dist = (stat1 - stat2)**2
            # weigh by the minimum, the weight should reflect the realiability of the difference
            # (if any of the two is low, its value, and thus the difference, is not reliable)
            # dist = dist * np.minimum(stat1_w, stat2_w) # HERE FOR WEIGHTED
            dists[stat_name] += dist.sum()
        elif stat_name in ['neighbor_type_hist']:
            dist = (stat1 - stat2)**2
            # weigh by the minimum, the weight should reflect the realiability of the difference
            # (if any of the two is low, its value, and thus the difference, is not reliable)
            # dist = dist * np.expand_dims(np.minimum(stat1_w, stat2_w), -1) # HERE FOR WEIGHTED
            dists[stat_name] += dist.sum()
        elif stat_name in ['center_x', 'center_y', 'area', 'aspect', 'neighbor_count_hist', 'exterior_dist']:
            # max_ind = stat1.shape[1]-1
            # stat_x = np.linspace(0, max_ind, max_ind+1)
            stat_x_val = stat_x[1] / (stat_x[1].max() - stat_x[1].min()) # normalized so max. distance between x values is 1

            w_total = 0
            for c in range(stat1.shape[0]):
                # weight: average number of rooms of this type in a floor plan (max. over stat1 and stat2)
                # w = max(type_freq1[c], type_freq2[c])
                w = max(stat1_w[c], stat2_w[c])

                # normalize with average room count of the current type
                if stat1_w[c] == 0 and stat2_w[c] == 0:
                    # both are 0, this does not add to the distance
                    pass
                elif stat1_w[c] == 0 or stat2_w[c] == 0:
                    # one of them is 0, add maximum distance of 1
                    w_total += w
                    dists[stat_name] += w
                # elif stat1[c].sum() == 0 or stat2[c].sum() == 0:
                #     # one of the distributions is all 0, probably because of , add maximum distance of 1 (weighted as usual)
                #     w_total += w
                #     dists[stat_name] += w
                else:
                    w_total += w
                    if stat1[c, :].sum() <= 0 or stat2[c, :].sum() <= 0:
                        dists[stat_name] += abs(stat2[c, :].sum() - stat1[c, :].sum())
                    else:
                        dists[stat_name] += w * sps.wasserstein_distance(
                            u_values=stat_x_val, v_values=stat_x_val, u_weights=stat1[c, :], v_weights=stat2[c, :])

            if w_total > 0:
                dists[stat_name] /= w_total
            else:
                dists[stat_name] = float(dists[stat_name])
        elif stat_name in [
                'center_x_dist', 'center_y_dist', 'gap', 'center_align_best', 'center_align_worst',
                'side_align_best', 'side_align_second_best', 'side_align_second_worst', 'side_align_worst',
                'neighbor_center_x_dist', 'neighbor_center_y_dist', 'neighbor_gap', 'neighbor_center_align_best', 'neighbor_center_align_worst',
                'neighbor_side_align_best', 'neighbor_side_align_second_best', 'neighbor_side_align_second_worst', 'neighbor_side_align_worst']:

            stat_x_val = stat_x / (stat_x.max() - stat_x.min()) # normalized so max. distance between x values is 1
            if stat1.sum() <= 0 or stat2.sum() <= 0:
                dists[stat_name] += abs(stat2.sum() - stat1.sum())
            else:
                dists[stat_name] += sps.wasserstein_distance(
                    u_values=stat_x_val, v_values=stat_x_val, u_weights=stat1, v_weights=stat2)
        else:
            raise ValueError(f'Unknown stat: {stat_name}')

    return dists

def save_stats(filename, stats):
    np.save(filename, stats)

def load_stats(filename):
    return np.load(filename, allow_pickle=True).item()

def vis_egraph_set_stat_dists(real_stats, fake_stats, stat_dists, filename=None):

    if isinstance(real_stats, str):
        real_stats = load_stats(filename=real_stats)
    if isinstance(fake_stats, str):
        fake_stats = load_stats(filename=fake_stats)
    if isinstance(stat_dists, str):
        stat_dists = load_stats(filename=stat_dists)

    label_names = [
        'Exterior',
        'Wall',
        'Kitchen',
        'Bedroom',
        'Bathroom',
        'Living Room',
        'Office',
        'Garage',
        'Balcony',
        'Hallway',
        'Other Room']

    for stat_name, fake_stat in fake_stats.items():
        stat_x = fake_stat['x']
        fake_stat = fake_stat['y']
        real_stat = real_stats[stat_name]['y']

        if stat_name == 'type_freq' or stat_name == 'unreachable':
            plt.figure()
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
            if stat_name == 'type_freq':
                title = 'Room type frequency'
            elif stat_name == 'unreachable':
                title = 'Unreachable room frequency by type'
            else:
                title = 'Unknown'
            fig.suptitle(title)

            ax.bar(x=stat_x, height=fake_stat, alpha=0.5, label='fake', width=(stat_x[1] - stat_x[0]) * 0.8)
            ax.bar(x=stat_x, height=real_stat, alpha=0.5, label='real', width=(stat_x[1] - stat_x[0]) * 0.8)
            ax.legend(loc='upper center')

            # figs[stat_name] = fig
            if filename is not None:
                fig.savefig(f'{filename}_{stat_name}.pdf', bbox_inches='tight')
                fig.savefig(f'{filename}_{stat_name}.svg', bbox_inches='tight')

            plt.close(fig=fig)

        elif stat_name in ['neighbor_type_hist', 'type_dist']:
            plt.figure()
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

            if stat_name == 'neighbor_type_hist':
                title = f'Neighbor type probabilities for a given type (in each row)'
            elif stat_name == 'type_dist':
                title = f'Average graph distances between room types'
            else:
                title = f'Unknown'
            if stat_name in stat_dists:
                title += f' (L2 distance: {stat_dists[stat_name]:.8})'
            fig.suptitle(title)

            ax[0].imshow(fake_stat)
            ax[0].set_title('fake')
            plt.sca(ax[0])
            plt.xticks(range(fake_stat.shape[1]), label_names, rotation=45, ha='right')
            plt.yticks(range(fake_stat.shape[0]), label_names, rotation=45)

            ax[1].imshow(real_stat)
            ax[1].set_title('real')
            plt.sca(ax[1])
            plt.xticks(range(real_stat.shape[1]), label_names, rotation=45, ha='right')
            plt.yticks(range(real_stat.shape[0]), label_names, rotation=45)

            # figs[stat_name] = fig
            if filename is not None:
                fig.savefig(f'{filename}_{stat_name}.pdf', bbox_inches='tight')
                fig.savefig(f'{filename}_{stat_name}.svg', bbox_inches='tight')

            plt.close(fig=fig)

        elif stat_name in ['type_hist', 'center_x', 'center_y', 'area', 'aspect', 'neighbor_count_hist', 'exterior_dist']:
            plt.figure()
            nrows = math.ceil(fake_stat.shape[0]/4.0)
            fig, ax = plt.subplots(nrows=nrows, ncols=4, figsize=(16, nrows*2))

            if stat_name == 'type_hist':
                title = f'Average room type count'
            elif stat_name == 'center_x':
                title = f'Room type center x distribution'
            elif stat_name == 'center_y':
                title = f'Room type center y distribution'
            elif stat_name == 'area':
                title = f'Room type area distribution'
            elif stat_name == 'aspect':
                title = f'Room type aspect ratio distribution'
            elif stat_name == 'neighbor_count_hist':
                title = f'Room type neighbor count distribution'
            elif stat_name == 'exterior_dist':
                title = f'Average distance to exterior from room types'
            else:
                title = f'Unknown'
            if stat_name in stat_dists:
                title += f' (average EM distance: {stat_dists[stat_name]:.8})'
            fig.suptitle(title)

            c = 0
            for rind in range(nrows):
                for cind in range(4):
                    if c < fake_stat.shape[0]:
                        ax[rind, cind].bar(x=stat_x[1], height=fake_stat[c, :], alpha=0.5, label='fake', width=(stat_x[1][1] - stat_x[1][0]) * 0.8)
                        ax[rind, cind].bar(x=stat_x[1], height=real_stat[c, :], alpha=0.5, label='real', width=(stat_x[1][1] - stat_x[1][0]) * 0.8)
                        ax[rind, cind].set_title(label_names[c])
                        if c == 0:
                            ax[rind, cind].legend(loc='upper center')
                    else:
                        fig.delaxes(ax[rind, cind])
                    c += 1
            fig.subplots_adjust(hspace=0.4)

            # figs[stat_name] = fig
            if filename is not None:
                fig.savefig(f'{filename}_{stat_name}.pdf', bbox_inches='tight')
                fig.savefig(f'{filename}_{stat_name}.svg', bbox_inches='tight')

            plt.close(fig=fig)

        elif stat_name in [
                'center_x_dist', 'center_y_dist', 'gap', 'center_align_best', 'center_align_worst',
                'side_align_best', 'side_align_second_best', 'side_align_second_worst', 'side_align_worst',
                'neighbor_center_x_dist', 'neighbor_center_y_dist', 'neighbor_gap', 'neighbor_center_align_best', 'neighbor_center_align_worst',
                'neighbor_side_align_best', 'neighbor_side_align_second_best', 'neighbor_side_align_second_worst', 'neighbor_side_align_worst']:

            plt.figure()
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))

            if stat_name == 'center_x_dist':
                title = f'Distance between center x coordinates for all room pairs'
            elif stat_name == 'center_y_dist':
                title = f'Distance between center y coordinates for all room pairs'
            elif stat_name == 'gap':
                title = f'Size of the axis-aligned gap between all room pairs'
            elif stat_name == 'center_align_best':
                title = f'Deviation from alignment for the most aligned room center coordinate for all room pairs'
            elif stat_name == 'center_align_worst':
                title = f'Deviation from alignment for the least aligned room center coordinate for all room pairs'
            elif stat_name == 'side_align_best':
                title = f'Deviation from alignment for the most aligned room side for all room pairs'
            elif stat_name == 'side_align_second_best':
                title = f'Deviation from alignment for the second most aligned room side for all room pairs'
            elif stat_name == 'side_align_second_worst':
                title = f'Deviation from alignment for the second least aligned room side for all room pairs'
            elif stat_name == 'side_align_worst':
                title = f'Deviation from alignment for the least aligned room side for all room pairs'
            elif stat_name == 'neighbor_center_x_dist':
                title = f'Distance between center x coordinates for neighboring rooms'
            elif stat_name == 'neighbor_center_y_dist':
                title = f'Distance between center y coordinates for neighboring rooms'
            elif stat_name == 'neighbor_gap':
                title = f'Size of the axis-aligned gap between neighboring rooms'
            elif stat_name == 'neighbor_center_align_best':
                title = f'Deviation from alignment for the most aligned room center coordinate for neighboring rooms'
            elif stat_name == 'neighbor_center_align_worst':
                title = f'Deviation from alignment for the least aligned room center coordinate for neighboring rooms'
            elif stat_name == 'neighbor_side_align_best':
                title = f'Deviation from alignment for the most aligned room side for neighboring rooms'
            elif stat_name == 'neighbor_side_align_second_best':
                title = f'Deviation from alignment for the second most aligned room side for neighboring rooms'
            elif stat_name == 'neighbor_side_align_second_worst':
                title = f'Deviation from alignment for the second least aligned room side for neighboring rooms'
            elif stat_name == 'neighbor_side_align_worst':
                title = f'Deviation from alignment for the least aligned room side for neighboring rooms'
            else:
                title = f'Unknown'
            if stat_name in stat_dists:
                title += f' (EM distance: {stat_dists[stat_name]:.8})'
            fig.suptitle(title)

            ax.bar(x=stat_x, height=fake_stat, alpha=0.5, label='fake', width=(stat_x[1] - stat_x[0]) * 0.8)
            ax.bar(x=stat_x, height=real_stat, alpha=0.5, label='real', width=(stat_x[1] - stat_x[0]) * 0.8)
            ax.legend(loc='upper center')

            # figs[stat_name] = fig
            if filename is not None:
                fig.savefig(f'{filename}_{stat_name}.pdf', bbox_inches='tight')
                fig.savefig(f'{filename}_{stat_name}.svg', bbox_inches='tight')

            plt.close(fig=fig)

        else:
            raise ValueError(f'Unknown stat name {stat_name}.')

    # return figs

def compute_egraph_set_stats(out_filename, input_dir, input_list, suffix, label_count, exclude_types):

    # egraph_names = []
    # with open(egraph_list_filename) as f:
    #     egraph_names = f.readlines()
    # egraph_names = [x.strip() for x in egraph_names]
    # egraph_names = list(filter(None, egraph_names))
    # egraph_names = [os.path.join(egraph_dir, gn+egraph_suffix) for gn in egraph_names]

    sample_names, sample_format = get_sample_names(input_dir=input_dir, input_list=input_list)

    set_stats = None
    for sample_name in tqdm(sample_names):
        boxes, door_edges, wall_edges, sample_names = load_boxes(input_dir=input_dir, sample_names=[sample_name], format=sample_format, suffix=suffix)

        try:
            room_types, room_bboxes, room_door_adj, room_masks = convert_boxes_to_rooms(
                boxes=boxes, door_edges=door_edges, wall_edges=wall_edges, img_res=(64, 64), room_type_count=label_count)
        except ValueError as err:
            print(f'WARNING: could not parse sample {sample_name}:\n{err}')
            continue

        node_type = room_types[0].unsqueeze(dim=0)
        pt = room_bboxes[0].to(dtype=torch.float32)
        a = room_door_adj[0].to(dtype=torch.float32)

        stats = egraph_stats(
            points=pt.unsqueeze(dim=0), point_type=node_type.unsqueeze(dim=0), adj=a.unsqueeze(dim=0),
            label_count=label_count, exclude_types=exclude_types)
        if set_stats is None:
            set_stats = stats
        else:
            for stat_name, stat in stats.items():
                set_stats[stat_name].extend(stat) # pylint: disable=E1136


    set_stats = egraph_set_stats(stats=set_stats, label_count=label_count)

    if out_filename is not None:
        if not os.path.exists(os.path.dirname(out_filename)):
            os.makedirs(os.path.dirname(out_filename))
        save_stats(filename=out_filename, stats=set_stats)

    return set_stats

# either provide real_stat_filename and fake_stat_filename,
# or all the arguments below these two to compute the stats
def compute_egraph_set_stat_dists(
        out_dirname,
        real_stat_filename=None, fake_stat_filename=None,
        real_egraph_dir=None, fake_egraph_dir=None,
        exclude_types=None, vis=True):

    if not os.path.exists(out_dirname):
        os.makedirs(out_dirname)

    if real_stat_filename is not None:
        print('loading stats of real egraph set ...')
        real_set_stats = load_stats(filename=real_stat_filename)
        save_stats(filename=os.path.join(out_dirname, 'stats_real.npy'), stats=real_set_stats)
    else:
        print('computing stats of real egraph set ...')
        real_set_stats = compute_egraph_set_stats(
            out_filename=os.path.join(out_dirname, 'stats_real.npy'), egraph_dir=real_egraph_dir,
            node_dim=node_dim, label_count=label_count, exclude_types=exclude_types)

    if fake_stat_filename is not None:
        print('loading stats of fake egraph set ...')
        fake_set_stats = load_stats(filename=fake_stat_filename)
        save_stats(filename=os.path.join(out_dirname, 'stats_fake.npy'), stats=fake_set_stats)
    else:
        print('computing stats of fake egraph set ...')
        fake_set_stats = compute_egraph_set_stats(
            out_filename=os.path.join(out_dirname, 'stats_fake.npy'), egraph_dir=fake_egraph_dir,
            node_dim=node_dim, label_count=label_count, exclude_types=exclude_types)

    print('computing stat difference ...')
    sds = egraph_set_stat_dists(stats1=fake_set_stats, stats2=real_set_stats)

    # save stats distance as npy file and distance summary as text file
    save_stats(filename=os.path.join(out_dirname, 'stat_dists.npy'), stats=sds)
    save_egraph_set_stat_dists(stat_dists=sds, filename=os.path.join(out_dirname, 'stat_dists.txt'))

    # visualize stat difference
    if vis:
        print('visualizing stat difference ...')
        vis_egraph_set_stat_dists(
            real_stats=real_set_stats, fake_stats=fake_set_stats, stat_dists=sds,
            filename=os.path.join(out_dirname, 'vis'))

def save_egraph_set_stat_dists(stat_dists, filename):
    topology_list = ['type_freq', 'type_hist', 'type_dist', 'neighbor_count_hist', 'neighbor_type_hist', 'unreachable', 'exterior_dist']

    spatial_single_list = ['center_x', 'center_y', 'area', 'aspect']

    spatial_pair_list = [
        'center_x_dist', 'center_y_dist', 'gap', 'center_align_best',
        'center_align_worst', 'side_align_best', 'side_align_second_best', 'side_align_second_worst',
        'side_align_worst', 'neighbor_center_x_dist', 'neighbor_center_y_dist', 'neighbor_gap', 'neighbor_center_align_best',
        'neighbor_center_align_worst', 'neighbor_side_align_best', 'neighbor_side_align_second_best', 'neighbor_side_align_second_worst',
        'neighbor_side_align_worst']

    with open(filename, 'w') as f:
        f.write('topology:\n')
        for stat_name in topology_list:
            if stat_name in stat_dists:
                f.write(f'{stat_name}: {stat_dists[stat_name]:.6f}\n')

        f.write('\n')
        f.write('spatial single:\n')
        for stat_name in spatial_single_list:
            if stat_name in stat_dists:
                f.write(f'{stat_name}: {stat_dists[stat_name]:.6f}\n')

        f.write('\n')
        f.write('spatial pair:\n')
        for stat_name in spatial_pair_list:
            if stat_name in stat_dists:
                f.write(f'{stat_name}: {stat_dists[stat_name]:.6f}\n')

if __name__ == '__main__':

    from convert_boxes_to_rooms import room_type_names

    exclude_types = [1]

    # input_dir = '../data/results/5_tuples_t_0.8'
    # stats_filename = '../data/results/5_tuples_t_0.8/stats.npy'
    # input_list = None
    # suffix = ''

    # input_dir = '/home/guerrero/scratch_space/floorplan/rplan_ddg_var'
    # stats_filename = '/home/guerrero/scratch_space/floorplan/stats.npy'
    # input_list = '/home/guerrero/scratch_space/floorplan/rplan_ddg_var/test.txt'
    # suffix='_image_nodoor'

    # input_dir = '../data/results/rplan_var_images_doors'
    # stats_filename = '../data/results/rplan_var_images_doors/stats.npy'
    # input_list = '../data/results/rplan_var_images_doors/all.txt'
    # suffix=''

    # compute_egraph_set_stats(
    #     out_filename=stats_filename, input_dir=input_dir, input_list=input_list, suffix=suffix, label_count=len(room_type_names), exclude_types=exclude_types)


    # GT
    real_stat_filename = '/home/guerrero/scratch_space/floorplan/stats.npy'

    # ours
    fake_stat_filename = '../data/results/5_tuples_t_0.8/stats.npy'
    out_dirname = '../data/results/5_tuples_t_0.8/stat_dists'

    # # RPLAN baseline
    # fake_stat_filename = '../data/results/rplan_var_images_doors/stats.npy'
    # out_dirname = '../data/results/rplan_var_images_doors_stat_dists'

    compute_egraph_set_stat_dists(
        out_dirname=out_dirname,
        real_stat_filename=real_stat_filename, fake_stat_filename=fake_stat_filename,
        exclude_types=exclude_types, vis=True)

    # # # ---------------------------------------------------
    # # # compare stats to gt
    # # # ---------------------------------------------------
    # # labels_filename = '../data/label_colors2.csv'
    # # evals = [
    # #     # # ours
    # #     # {'real': '/home/code-base/scratch_space/interiorgen/data/suncg_ddg_013/stats/stats_test.npy',
    # #     #  'fake': '../data/results/stage_graph/egraph_edge_cgan_011b_suncg__suncg_001/1084720/stats/stats_all.npy',
    # #     #  'real_name': 'orig'},
    # #     # {'real': '/home/code-base/scratch_space/interiorgen/data/suncg_ddg_013_parsedgraphs/stats/stats_test.npy',
    # #     #  'fake': '../data/results/stage_graph/egraph_edge_cgan_011b_suncg__suncg_001/1084720/stats/stats_all.npy',
    # #     #  'real_name': 'parsed'},
    # #     {'real': '/home/code-base/scratch_space/interiorgen/data/lifull_ddg_var/stats/stats_test.npy',
    # #      'fake': '../data/results/stage_graph/egraph_edge_cgan_011b_lifull_var__lifull_var_001/1085135/stats/stats_all.npy',
    # #      'real_name': 'orig'},
    # #     {'real': '/home/code-base/scratch_space/interiorgen/data/lifull_ddg_var_parsedgraphs/stats/stats_test.npy',
    # #      'fake': '../data/results/stage_graph/egraph_edge_cgan_011b_lifull_var__lifull_var_001/1085135/stats/stats_all.npy',
    # #      'real_name': 'parsed'},
    # #     {'real': '/home/code-base/scratch_space/interiorgen/data/rplan_ddg_var/stats/stats_test.npy',
    # #      'fake': '../data/results/stage_graph/egraph_edge_cgan_011b_rplan_var__rplan_var_001/600727/stats/stats_all.npy',
    # #      'real_name': 'orig'},
    # #     {'real': '/home/code-base/scratch_space/interiorgen/data/rplan_ddg_var_parsedgraphs/stats/stats_test.npy',
    # #      'fake': '../data/results/stage_graph/egraph_edge_cgan_011b_rplan_var__rplan_var_001/600727/stats/stats_all.npy',
    # #      'real_name': 'parsed'},

    # #     # # stylegan baseline
    # #     # {'real': '/home/code-base/scratch_space/interiorgen/data/suncg_ddg_013/stats/stats_test.npy',
    # #     #  'fake': '../data/results/stylegan/suncg_002_stylegan_baseline/stats/stats_all.npy',
    # #     #  'real_name': 'orig'},
    # #     # {'real': '/home/code-base/scratch_space/interiorgen/data/suncg_ddg_013_parsedgraphs/stats/stats_test.npy',
    # #     #  'fake': '../data/results/stylegan/suncg_002_stylegan_baseline/stats/stats_all.npy',
    # #     #  'real_name': 'parsed'},
    # #     # {'real': '/home/code-base/scratch_space/interiorgen/data/lifull_ddg_var/stats/stats_test.npy',
    # #     #  'fake': '../data/results/stylegan/lifull_005/stats/stats_all.npy',
    # #     #  'real_name': 'orig'},
    # #     # {'real': '/home/code-base/scratch_space/interiorgen/data/lifull_ddg_var_parsedgraphs/stats/stats_test.npy',
    # #     #  'fake': '../data/results/stylegan/lifull_005/stats/stats_all.npy',
    # #     #  'real_name': 'parsed'},
    # #     # {'real': '/home/code-base/scratch_space/interiorgen/data/rplan_ddg_var/stats/stats_test.npy',
    # #     #  'fake': '../data/results/stylegan/rplan_002_stylegan_baseline/stats/stats_all.npy',
    # #     #  'real_name': 'orig'},
    # #     # {'real': '/home/code-base/scratch_space/interiorgen/data/rplan_ddg_var_parsedgraphs/stats/stats_test.npy',
    # #     #  'fake': '../data/results/stylegan/rplan_002_stylegan_baseline/stats/stats_all.npy',
    # #     #  'real_name': 'parsed'},

    # #     # # pix2pixhd baseline
    # #     # {'real': '/home/code-base/scratch_space/interiorgen/data/suncg_ddg_013/stats/stats_test.npy',
    # #     #  'fake': '../data/results/pix2pixhd/suncg_ddg_001/images/stats/stats_test.npy',
    # #     #  'real_name': 'orig'},
    # #     # {'real': '/home/code-base/scratch_space/interiorgen/data/suncg_ddg_013_parsedgraphs/stats/stats_test.npy',
    # #     #  'fake': '../data/results/pix2pixhd/suncg_ddg_001/images/stats/stats_test.npy',
    # #     #  'real_name': 'parsed'},
    # #     # {'real': '/home/code-base/scratch_space/interiorgen/data/suncg_ddg_013/stats/stats_test.npy',
    # #     #  'fake': '../data/results/pix2pixhd/suncg_ddg_001_suncg_001/images/stats/stats_all.npy',
    # #     #  'real_name': 'orig'},
    # #     # {'real': '/home/code-base/scratch_space/interiorgen/data/suncg_ddg_013_parsedgraphs/stats/stats_test.npy',
    # #     #  'fake': '../data/results/pix2pixhd/suncg_ddg_001_suncg_001/images/stats/stats_all.npy',
    # #     #  'real_name': 'parsed'},
    # #     # {'real': '/home/code-base/scratch_space/interiorgen/data/lifull_ddg_var/stats/stats_test.npy',
    # #     #  'fake': '../data/results/pix2pixhd/lifull_ddg_001/images/stats/stats_test.npy',
    # #     #  'real_name': 'orig'},
    # #     # {'real': '/home/code-base/scratch_space/interiorgen/data/lifull_ddg_var_parsedgraphs/stats/stats_test.npy',
    # #     #  'fake': '../data/results/pix2pixhd/lifull_ddg_001/images/stats/stats_test.npy',
    # #     #  'real_name': 'parsed'},
    # #     # {'real': '/home/code-base/scratch_space/interiorgen/data/lifull_ddg_var/stats/stats_test.npy',
    # #     #  'fake': '../data/results/pix2pixhd/lifull_ddg_var_001_orig_lifull_var_001_epoch_20/images/stats/stats_all.npy',
    # #     #  'real_name': 'orig'},
    # #     # {'real': '/home/code-base/scratch_space/interiorgen/data/lifull_ddg_var_parsedgraphs/stats/stats_test.npy',
    # #     #  'fake': '../data/results/pix2pixhd/lifull_ddg_var_001_orig_lifull_var_001_epoch_20/images/stats/stats_all.npy',
    # #     #  'real_name': 'parsed'},
    # #     # {'real': '/home/code-base/scratch_space/interiorgen/data/rplan_ddg_var/stats/stats_test.npy',
    # #     #  'fake': '../data/results/pix2pixhd/rplan_ddg_001/images/stats/stats_test.npy',
    # #     #  'real_name': 'orig'},
    # #     # {'real': '/home/code-base/scratch_space/interiorgen/data/rplan_ddg_var_parsedgraphs/stats/stats_test.npy',
    # #     #  'fake': '../data/results/pix2pixhd/rplan_ddg_001/images/stats/stats_test.npy',
    # #     #  'real_name': 'parsed'},
    # #     # {'real': '/home/code-base/scratch_space/interiorgen/data/rplan_ddg_var/stats/stats_test.npy',
    # #     #  'fake': '../data/results/pix2pixhd/rplan_ddg_var_001_orig_rplan_var_001_epoch_20/images/stats/stats_all.npy',
    # #     #  'real_name': 'orig'},
    # #     # {'real': '/home/code-base/scratch_space/interiorgen/data/rplan_ddg_var_parsedgraphs/stats/stats_test.npy',
    # #     #  'fake': '../data/results/pix2pixhd/rplan_ddg_var_001_orig_rplan_var_001_epoch_20/images/stats/stats_all.npy',
    # #     #  'real_name': 'parsed'},

    # #     # # # rplan baseline
    # #     # {'real': '/home/code-base/scratch_space/interiorgen/data/lifull_ddg_var/stats/stats_test.npy',
    # #     #  'fake': '../data/results/rplan/stage_ddg/lifull_parsedgraphs/stats/stats_all.npy',
    # #     #  'real_name': 'orig'},
    # #     # {'real': '/home/code-base/scratch_space/interiorgen/data/lifull_ddg_var_parsedgraphs/stats/stats_test.npy',
    # #     #  'fake': '../data/results/rplan/stage_ddg/lifull_parsedgraphs/stats/stats_all.npy',
    # #     #  'real_name': 'parsed'},
    # #     # {'real': '/home/code-base/scratch_space/interiorgen/data/rplan_ddg_var/stats/stats_test.npy',
    # #     #  'fake': '../data/results/rplan/stage_ddg/rplan_parsedgraphs/stats/stats_all.npy',
    # #     #  'real_name': 'orig'},
    # #     # {'real': '/home/code-base/scratch_space/interiorgen/data/rplan_ddg_var_parsedgraphs/stats/stats_test.npy',
    # #     #  'fake': '../data/results/rplan/stage_ddg/rplan_parsedgraphs/stats/stats_all.npy',
    # #     #  'real_name': 'parsed'},
    # #     ]


    # # # for eval_info in evals:
    # # #     eval_info['stats_dirname'] = os.path.normpath(
    # # #         os.path.join(os.path.dirname(eval_info['fake']), '../stats_vs_'+eval_info['real_name']))
    # # # for eval_info in evals:
    # # #     print('------------------------')
    # # #     print(f'evaluating\n{eval_info["real"]}\nvs.\n{eval_info["fake"]}')
    # # #     print(f'and saving in\n{eval_info["stats_dirname"]}')
    # # #     print('------------------------')
    # # #     compute_egraph_set_stat_dists(
    # # #         out_dirname=eval_info['stats_dirname'],
    # # #         labels_filename=labels_filename,
    # # #         real_stat_filename=eval_info['real'],
    # # #         fake_stat_filename=eval_info['fake'])

    # # for eval_info in evals:
    # #     eval_info['stats_dirname'] = os.path.normpath(
    # #         os.path.join(os.path.dirname(eval_info['fake']), '../stats_vs_'+eval_info['real_name']))
    # # for eval_info in evals:
    # #     print('------------------------')
    # #     print(f'visualizing\n{eval_info["real"]}\nvs.\n{eval_info["fake"]}')
    # #     print(f'and saving in\n{eval_info["stats_dirname"]}')
    # #     print('------------------------')
    # #     vis_egraph_set_stat_dists(
    # #         real_stats=eval_info['real'], fake_stats=eval_info['fake'],
    # #         stat_dists=os.path.join(eval_info['stats_dirname'], 'stat_dists.npy'),
    # #         labels_filename=labels_filename,
    # #         filename=os.path.join(eval_info['stats_dirname'], 'vis'))


    # # ---------------------------------------------------
    # # compute stats
    # # ---------------------------------------------------
    # # compute stats without comparing
    # # unused, patio, window, door, wall
    # import argparse
    # from options import add_train_options
    # egraph_suffix = '_graph.npy'

    # # floorplans
    # exclude_types = [0, 2, 3, 4, 5]
    # labels_filename = '../data/label_colors2.csv'
    # label_offset = -1
    # label_count = 15
    # node_dim = 4
    # node_feature_scaling = 0.01
    # evals = [
    #     # # ground truth original and parsed graphs
    #     # {'dataset': '/home/code-base/scratch_space/interiorgen/data/suncg_ddg_013/test.txt'},
    #     # {'dataset': '/home/code-base/scratch_space/interiorgen/data/suncg_ddg_013_parsedgraphs/test.txt'},
    #     # {'dataset': '/scratch_space/ddg/lifull_ddg_var/test.txt'},
    #     # {'dataset': '/home/code-base/scratch_space/interiorgen/data/lifull_ddg_var_parsedgraphs/test.txt'},
    #     {'dataset': '/scratch_space/ddg/rplan_ddg_var/test.txt'},
    #     # {'dataset': '/home/code-base/scratch_space/interiorgen/data/rplan_ddg_var_parsedgraphs/test.txt'},

    #     # # generated graphs without doors
    #     # {'dataset': '../data/results/stage_image/suncg_001/all.txt'},
    #     # {'dataset': '../data/results/stage_image/rplan_001/all.txt'},
    #     # {'dataset': '../data/results/stage_image/lifull_004/all.txt'},

    #     # our graphs
    #     # {'dataset': '../data/results/stage_graph/egraph_edge_cgan_011b_suncg__suncg_001/1084720/all.txt'},
    #     # {'dataset': '../data/results/stage_graph/egraph_edge_cgan_011b_lifull__lifull_004/469455/all.txt',
    #     #  'add_excludes': [10]}, # also exclude offices
    #     # {'dataset': '../data/results/stage_graph/egraph_edge_cgan_011b_rplan__rplan_001/600727/all.txt'},

    #     # # stylegan baseline
    #     # {'dataset': '../data/results/stylegan/rplan_002_stylegan_baseline/all.txt'},
    #     # {'dataset': '../data/results/stylegan/suncg_002_stylegan_baseline/all.txt'},
    #     # {'dataset': '../data/results/stylegan/lifull_005/all.txt'},

    #     # # pix2pixhd baseline
    #     # {'dataset': '../data/results/pix2pixhd/suncg_ddg_001/images/test.txt'},
    #     # {'dataset': '../data/results/pix2pixhd/suncg_ddg_001_suncg_001/images/all.txt'},
    #     # {'dataset': '../data/results/pix2pixhd/lifull_ddg_001/images/test.txt'},
    #     # {'dataset': '../data/results/pix2pixhd/lifull_ddg_var_001_orig_lifull_var_001/images/all.txt'},
    #     # {'dataset': '../data/results/pix2pixhd/rplan_ddg_001/images/test.txt'},
    #     # {'dataset': '../data/results/pix2pixhd/rplan_ddg_001_rplan_001/images/all.txt'},
    # ]

    # # # rooms
    # # exclude_types = [0, 2, 3, 4, 5]
    # # labels_filename = '../data/furn_colors_3.csv'
    # # label_offset = 0
    # # label_count = 27
    # # node_dim = 4
    # # node_feature_scaling = 0.083333 # 1/12 meters
    # # evals = [
    # #     {'dataset': '/home/code-base/scratch_space/interiorgen/data/furniture_016/test.txt'},
    # # ]

    # for eval_info in evals:
    #     eval_info['stats_filename'] = os.path.join(
    #         os.path.dirname(eval_info['dataset']), 'stats',
    #         f'stats_{os.path.splitext(os.path.basename(eval_info["dataset"]))[0]}.npy')

    # for eval_info in evals:
    #     print('------------------------')
    #     print(f'evaluating {eval_info["dataset"]}')
    #     print('------------------------')

    #     eval_exclude_types = exclude_types
    #     if 'add_excludes' in eval_info:
    #         eval_exclude_types.extend(eval_info['add_excludes'])

    #     compute_egraph_set_stats(
    #         out_filename=eval_info['stats_filename'],
    #         egraph_dir=os.path.dirname(eval_info['dataset']),
    #         egraph_list_filename=eval_info['dataset'],
    #         egraph_suffix=egraph_suffix,
    #         node_dim=node_dim,
    #         node_mean=torch.zeros(1, 4), node_std=torch.ones(1, 4),
    #         node_feature_scaling=node_feature_scaling,
    #         label_count=label_count, label_offset=label_offset, labels_filename=labels_filename,
    #         exclude_types=eval_exclude_types)
