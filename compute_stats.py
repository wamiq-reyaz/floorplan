import os
import copy
import math
import numpy as np
import scipy.sparse.csgraph as csg
import scipy.stats as sps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'figure.max_open_warning': 0}) # to avoid warnings that too many plots are open (apparently this also counts number of open axes)
# from dataset import load_label_colors, load_egraph
# from options import import_legacy_train_options

from load_boxes import get_room_sample_names, load_rooms
# from convert_boxes_to_rooms import convert_boxes_to_rooms

# compute egraph statistics
def egraph_stats(node_type, nodes, edges, label_count, exterior_type, exclude_types=None):

    if node_type is None:
        raise ValueError('Need node type.')

    stats = {}
    if node_type is not None:
        stats['type_count'] = []

    if nodes is not None:
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

    if edges is not None:
        stats['neighbor_count_hist'] = []
        stats['neighbor_type_hist'] = []
        stats['unreachable'] = []
        stats['exterior_dist'] = []
        stats['type_dist'] = []

    if nodes is not None and edges is not None:
        stats['neighbor_center_x_dist'] = []
        stats['neighbor_center_y_dist'] = []
        stats['neighbor_gap'] = []
        stats['neighbor_center_align_best'] = []
        stats['neighbor_center_align_worst'] = []
        stats['neighbor_side_align_best'] = []
        stats['neighbor_side_align_second_best'] = []
        stats['neighbor_side_align_second_worst'] = []
        stats['neighbor_side_align_worst'] = []

    batch_size = len(node_type) if node_type is not None else (len(nodes) if nodes is not None else len(edges))
    for b in range(batch_size):
        nt = node_type[b].copy()
        n = nodes[b].copy().astype(np.float32) if nodes is not None else None
        e = edges[b].copy() if edges is not None else None

        if (nt is not None and nt.ndim != 1) or (n is not None and n.ndim != 2) or (e is not None and e.ndim != 2):
            raise ValueError('Incorrect shape for nodes or edges.')

        # check for empty floor plans
        if nt.size == 0:
            print('** Warning: skipping a floor plan without rooms. **')
            continue

        # create adjacency matrix from edge list
        if e is not None:
            a = np.zeros([nt.size, nt.size], dtype=np.bool)
            a[e[:, 0], e[:, 1]] = True
            a = a | a.transpose() # symmetrize
        else:
            a = None

        if exclude_types is not None:
            include_mask = np.all([nt != et for et in exclude_types], axis=0)
            nt = nt[include_mask]
            n = n[include_mask, :] if n is not None else None
            a = a[include_mask, :][:, include_mask] if a is not None else None

        exterior_mask = nt == exterior_type

        if exterior_mask.all():
            print('** Warning: skipping a floor plan with only exterior. **')
            continue

        if n is not None:
            min_x = n[:, 0]
            min_y = n[:, 1]
            w = n[:, 2]
            h = n[:, 3]

            max_x = min_x + w
            max_y = min_y + h
            center_x = min_x + w/2.0
            center_y = min_y + h/2.0

        if nt is not None:
            type_bin_centers, type_bin_edges = hist_bins_uniform(0, label_count-1, label_count)

            stats['type_count'].append({
                'x': type_bin_centers,
                'y': np.histogram(nt, bins=type_bin_edges)[0]})

        if n is not None:

            pair_mask = np.ones([n.shape[0], n.shape[0]], dtype=bool)
            pair_mask[np.tril_indices(pair_mask.shape[0], -1)] = False
            # exclude exterior from pairs
            pair_mask[:, exterior_mask] = False
            pair_mask[exterior_mask, :] = False
            num_pairs = pair_mask.sum()

            val_bin_centers, val_bin_edges = hist_bins_uniform(0, 63, 21, negative_overflow=True, positive_overflow=True)

            stats['center_x'].append({
                'x': (type_bin_centers, val_bin_centers),
                'y': np.histogram2d(nt, center_x, bins=[type_bin_edges, val_bin_edges])[0]})
            stats['center_y'].append({
                'x': (type_bin_centers, val_bin_centers),
                'y': np.histogram2d(nt, center_y, bins=[type_bin_edges, val_bin_edges])[0]})

            val_bin_centers, val_bin_edges = hist_bins_uniform(0, 1000, 21, negative_overflow=True, positive_overflow=True)
            stats['area'].append({
                'x': (type_bin_centers, val_bin_centers),
                'y': np.histogram2d(nt, w * h, bins=[type_bin_edges, val_bin_edges])[0]})

            val_bin_centers, val_bin_edges = hist_bins_uniform(0, 1, 21, negative_overflow=True, positive_overflow=True)
            max_len = n[:, [2, 3]].max(axis=1)
            min_len = n[:, [2, 3]].min(axis=1)
            aspect = np.zeros(max_len.shape)
            mask = max_len > 0
            aspect[mask] = min_len[mask] / max_len[mask]
            stats['aspect'].append({
                'x': (type_bin_centers, val_bin_centers),
                'y': np.histogram2d(nt, aspect, bins=[type_bin_edges, val_bin_edges])[0]})

            # axis-aligned center distance
            val_bin_centers, val_bin_edges = hist_bins_uniform(0, 40, 41, negative_overflow=False, positive_overflow=True)
            offsets = center_x.reshape(1, -1) - center_x.reshape(-1, 1)
            offsets = np.absolute(offsets[pair_mask])
            stats['center_x_dist'].append({
                'x': val_bin_centers,
                'y': np.histogram(offsets, bins=val_bin_edges)[0] / num_pairs})
            offsets = center_y.reshape(1, -1) - center_y.reshape(-1, 1)
            offsets = np.absolute(offsets[pair_mask])
            stats['center_y_dist'].append({
                'x': val_bin_centers,
                'y': np.histogram(offsets, bins=val_bin_edges)[0] / num_pairs})

            # axis-aligned gap
            # (this gap definition contiues shrinking as a smaller object moves into a larger one)
            val_bin_centers, val_bin_edges = hist_bins_uniform(-10, 10, 21, negative_overflow=True, positive_overflow=True)
            gap_x = np.maximum(
                min_x.reshape(1, -1) - max_x.reshape(-1, 1),
                min_x.reshape(-1, 1) - max_x.reshape(1, -1))
            gap_y = np.maximum(
                min_y.reshape(1, -1) - max_y.reshape(-1, 1),
                min_y.reshape(-1, 1) - max_y.reshape(1, -1))
            gap = np.maximum(gap_x, gap_y)
            gap = gap[pair_mask]
            stats['gap'].append({
                'x': val_bin_centers,
                'y': np.histogram(gap, bins=val_bin_edges)[0] / num_pairs})

            # alignment
            val_bin_centers, val_bin_edges = hist_bins_uniform(0, 40, 41, negative_overflow=False, positive_overflow=True)
            dist_center_x = np.absolute((center_x.reshape(1, -1) - center_x.reshape(-1, 1))[pair_mask])
            dist_center_y = np.absolute((center_y.reshape(1, -1) - center_y.reshape(-1, 1))[pair_mask])
            dist_center = np.stack([dist_center_x, dist_center_y])
            stats['center_align_best'].append({
                'x': val_bin_centers,
                'y': np.histogram(dist_center.min(axis=0), bins=val_bin_edges)[0] / num_pairs})
            stats['center_align_worst'].append({
                'x': val_bin_centers,
                'y': np.histogram(dist_center.max(axis=0), bins=val_bin_edges)[0] / num_pairs})

            dist_side_left = np.absolute((min_x.reshape(1, -1) - min_x.reshape(-1, 1))[pair_mask])
            dist_side_right = np.absolute((max_x.reshape(1, -1) - max_x.reshape(-1, 1))[pair_mask])
            dist_side_bottom = np.absolute((min_y.reshape(1, -1) - min_y.reshape(-1, 1))[pair_mask])
            dist_side_top = np.absolute((max_y.reshape(1, -1) - max_y.reshape(-1, 1))[pair_mask])
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
                'y': np.histogram2d(nt, a.sum(axis=0), bins=[type_bin_edges, val_bin_edges])[0]})

            idx1, idx2 = np.nonzero(a)
            stats['neighbor_type_hist'].append({
                'x': (type_bin_centers, type_bin_centers),
                'y': np.histogram2d(nt[idx1], nt[idx2], bins=[type_bin_edges]*2)[0]})

            val_bin_centers, val_bin_edges = hist_bins_uniform(0, 10, 11, negative_overflow=False, positive_overflow=True)
            graph = csg.csgraph_from_dense(a)
            gdists = csg.shortest_path(graph, directed=False, unweighted=True)
            exterior_inds, = np.nonzero(nt == exterior_type)
            if exterior_inds.size > 0:
                exterior_dists = np.min(gdists[exterior_inds, :], axis=0)
            else:
                exterior_dists = np.full(shape=[gdists.shape[1]], fill_value=np.inf)
            unreachable_mask = exterior_dists == np.inf
            stats['unreachable'].append({
                'x': type_bin_centers,
                'y': np.histogram(nt[unreachable_mask], bins=type_bin_edges)[0]})
            reachable_mask = np.logical_not(unreachable_mask)
            stats['exterior_dist'].append({
                'x': (type_bin_centers, val_bin_centers),
                'y': np.histogram2d(nt[reachable_mask], exterior_dists[reachable_mask], bins=[type_bin_edges, val_bin_edges])[0]})

            reachable_mask = gdists != np.inf
            idx1, idx2 = np.nonzero(reachable_mask)
            stats['type_dist'].append({
                'x': (type_bin_centers, type_bin_centers, val_bin_centers),
                'y': np.histogramdd([nt[idx1], nt[idx2], gdists[reachable_mask]], bins=[type_bin_edges, type_bin_edges, val_bin_edges])[0]})

        if n is not None and a is not None:

            pair_mask = a.copy()
            pair_mask[np.tril_indices(pair_mask.shape[0], -1)] = False
            # exclude exterior from pairs
            pair_mask[:, exterior_mask] = False
            pair_mask[exterior_mask, :] = False
            num_pairs = pair_mask.sum()

            # axis-aligned center distance
            val_bin_centers, val_bin_edges = hist_bins_uniform(0, 40, 41, negative_overflow=False, positive_overflow=True)
            offsets = center_x.reshape(1, -1) - center_x.reshape(-1, 1)
            offsets = np.absolute(offsets[pair_mask])
            stats['neighbor_center_x_dist'].append({
                'x': val_bin_centers,
                'y': np.histogram(offsets, bins=val_bin_edges)[0] / (num_pairs if num_pairs > 0 else 1)})
            offsets = center_y.reshape(1, -1) - center_y.reshape(-1, 1)
            offsets = np.absolute(offsets[pair_mask])
            stats['neighbor_center_y_dist'].append({
                'x': val_bin_centers,
                'y': np.histogram(offsets, bins=val_bin_edges)[0] / (num_pairs if num_pairs > 0 else 1)})

            # axis-aligned gap
            # (this gap definition contiues shrinking as a smaller object moves into a larger one)
            val_bin_centers, val_bin_edges = hist_bins_uniform(-10, 10, 21, negative_overflow=True, positive_overflow=True)
            gap_x = np.maximum(
                min_x.reshape(1, -1) - max_x.reshape(-1, 1),
                min_x.reshape(-1, 1) - max_x.reshape(1, -1))
            gap_y = np.maximum(
                min_y.reshape(1, -1) - max_y.reshape(-1, 1),
                min_y.reshape(-1, 1) - max_y.reshape(1, -1))
            gap = np.maximum(gap_x, gap_y)
            gap = gap[pair_mask]
            stats['neighbor_gap'].append({
                'x': val_bin_centers,
                'y': np.histogram(gap, bins=val_bin_edges)[0] / (num_pairs if num_pairs > 0 else 1)})

            # alignment
            val_bin_centers, val_bin_edges = hist_bins_uniform(0, 40, 41, negative_overflow=False, positive_overflow=True)
            dist_center_x = np.absolute((center_x.reshape(1, -1) - center_x.reshape(-1, 1))[pair_mask])
            dist_center_y = np.absolute((center_y.reshape(1, -1) - center_y.reshape(-1, 1))[pair_mask])
            dist_center = np.stack([dist_center_x, dist_center_y])
            stats['neighbor_center_align_best'].append({
                'x': val_bin_centers,
                'y': np.histogram(dist_center.min(axis=0), bins=val_bin_edges)[0] / (num_pairs if num_pairs > 0 else 1)})
            stats['neighbor_center_align_worst'].append({
                'x': val_bin_centers,
                'y': np.histogram(dist_center.max(axis=0), bins=val_bin_edges)[0] / (num_pairs if num_pairs > 0 else 1)})

            dist_side_left = np.absolute((min_x.reshape(1, -1) - min_x.reshape(-1, 1))[pair_mask])
            dist_side_right = np.absolute((max_x.reshape(1, -1) - max_x.reshape(-1, 1))[pair_mask])
            dist_side_bottom = np.absolute((min_y.reshape(1, -1) - min_y.reshape(-1, 1))[pair_mask])
            dist_side_top = np.absolute((max_y.reshape(1, -1) - max_y.reshape(-1, 1))[pair_mask])
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

    egraph_stats_norm = {}
    # normalize set stats given knowledge about the entire set
    sample_count = None
    for stat_name, stat in stats.items():

        # stack into single ndarray
        stat_x = stat[0]['x'] # all should have the same x
        stat = np.stack([s['y'] for s in stat])
        if sample_count is None:
            sample_count = stat.shape[0]
        else:
            if stat.shape[0] != sample_count:
                raise ValueError('Inconsistent sample counts in different stats.')

        if stat_name == 'type_count':
            type_bin_centers, type_bin_edges = hist_bins_uniform(0, label_count-1, label_count)
            val_bin_centers, val_bin_edges = hist_bins_uniform(0, 40, 41, negative_overflow=False, positive_overflow=True)
            _, type_idx = np.indices(stat.shape)
            # floorplan statistics
            # each histogram is (n_types)
            type_idx = stat_x[type_idx]

            egraph_stats_norm['type_freq'] = {'x': stat_x, 'y': stat}

            stat = np.stack([np.histogram2d(type_idx[i], stat[i], bins=[type_bin_edges, val_bin_edges])[0] for i in range(type_idx.shape[0])])

            egraph_stats_norm['type_hist'] = {
                'x': (type_bin_centers, val_bin_centers),
                'y': stat}
        elif stat_name == 'unreachable':
            # floorplan statistics
            # each histogram is (n_types)
            egraph_stats_norm[stat_name] = {'x': stat_x, 'y': stat}
        elif stat_name in ['center_x', 'center_y', 'area', 'aspect', 'neighbor_count_hist', 'neighbor_type_hist', 'exterior_dist']:
            # single room statistics
            # each histogram is (n_types x n_value_bins)
            # weight: mass distribution of the histogram among room types (that is factored out with normalize_histograms)

            # neighbor_type_hist is the distribution of neighboring room types for each starting room type.
            # This is normalized to sum up to one per starting room type here (each row in the vis. sums to 1)

            norm_fac = stat.mean(axis=0).sum(axis=1) # normalize by total count per node type (sum over counts in all bins for each type)
            mask = norm_fac > 0
            stat[:, mask, :] = stat[:, mask, :] / norm_fac.reshape(1, -1, 1)[:, mask, :] # take average over all distances (from all floor plans) in each type pair
            stat[:, ~mask, :] = 0

            egraph_stats_norm[stat_name] = {
                'x': stat_x,
                'y': stat,
                'weight': norm_fac}
        elif stat_name in [
                'center_x_dist', 'center_y_dist', 'gap', 'center_align_best', 'center_align_worst',
                'side_align_best', 'side_align_second_best', 'side_align_second_worst', 'side_align_worst',
                'neighbor_center_x_dist', 'neighbor_center_y_dist', 'neighbor_gap', 'neighbor_center_align_best', 'neighbor_center_align_worst',
                'neighbor_side_align_best', 'neighbor_side_align_second_best', 'neighbor_side_align_second_worst', 'neighbor_side_align_worst']:
            # room pair statistics (have already been normalized over room pairs in each floor plan)
            # each histogram is (n_value_bins)
            egraph_stats_norm[stat_name] = {'x': stat_x, 'y': stat}
        elif stat_name == 'type_dist':
            # room pair statistics per type
            # each histogram is (n_types x n_types x n_distance_bins)

            norm_fac = stat.mean(axis=0).sum(axis=2) # normalize by number of pairs for each node type
            stat = (stat * stat_x[2]).sum(axis=-1)
            mask = norm_fac > 0.0001
            stat[:, mask] = stat[:, mask] / norm_fac[mask] # take average over all distances (from all floor plans) in each type pair
            stat[:, ~mask] = 0

            egraph_stats_norm[stat_name] = {
                'x': stat_x,
                'y': stat,
                'weight': norm_fac}
        else:
            raise ValueError(f'Unknown stat: {stat_name}')

    if sample_count < 2:
        raise ValueError('Must have at least two samples to compute set statistics.')

    set_stats = {}
    for stat_name, stat in egraph_stats_norm.items():

        # compute average
        set_stats[stat_name] = {
            'x': stat['x'],
            'y': stat['y'].mean(axis=0)} # mean
        if 'weight' in stat:
            set_stats[stat_name]['weight'] = stat['weight']

        # compute standard deviation
        set_stats[stat_name]['std'] = 0
        for si in range(sample_count):
           sample_stat = {'x': stat['x'], 'y': stat['y'][si].astype(np.float64)}
           if 'weight' in stat:
               sample_stat['weight'] =  stat['weight']
           # accumulate squared distance between sample and set average
           set_stats[stat_name]['std'] += egraph_set_single_stat_dist(sample_stat, set_stats[stat_name], stat_name)**2
        set_stats[stat_name]['std'] = np.sqrt(set_stats[stat_name]['std'] / (sample_count-1))

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

def egraph_set_single_stat_dist(stat1, stat2, stat_name):

    stat_x = stat1['x']
    if 'weight' in stat1:
        stat1_w = stat1['weight']
        stat2_w = stat2['weight']
    else:
        stat1_w = None
        stat2_w = None
    stat1 = stat1['y']
    stat2 = stat2['y']

    dist = 0

    if stat_name in ['type_freq', 'unreachable', 'type_dist', 'neighbor_type_hist']:
        dist += ((stat1 - stat2)**2).sum()
    elif stat_name == 'type_hist':
        # max_ind = stat1.shape[1]-1
        # stat_x = np.linspace(0, max_ind, max_ind+1)
        stat_x_val = stat_x[1] / (stat_x[1].max() - stat_x[1].min()) # normalized so max. distance between x values is 1

        for c in range(stat1.shape[0]):
            if stat1[c, :].sum() <= 0 or stat2[c, :].sum() <= 0:
                dist += abs(stat2[c, :].sum() - stat1[c, :].sum())
            else:
                dist += sps.wasserstein_distance(
                    u_values=stat_x_val, v_values=stat_x_val, u_weights=stat1[c, :], v_weights=stat2[c, :])

        dist /= stat1.shape[0]
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
                dist += w
            # elif stat1[c].sum() == 0 or stat2[c].sum() == 0:
            #     # one of the distributions is all 0, probably because of , add maximum distance of 1 (weighted as usual)
            #     w_total += w
            #     dist += w
            else:
                w_total += w
                if stat1[c, :].sum() <= 0 or stat2[c, :].sum() <= 0:
                    dist += abs(stat2[c, :].sum() - stat1[c, :].sum())
                else:
                    dist += w * sps.wasserstein_distance(
                        u_values=stat_x_val, v_values=stat_x_val, u_weights=stat1[c, :], v_weights=stat2[c, :])

        if w_total > 0:
            dist /= w_total
        else:
            dist = float(dist)
    elif stat_name in [
            'center_x_dist', 'center_y_dist', 'gap', 'center_align_best', 'center_align_worst',
            'side_align_best', 'side_align_second_best', 'side_align_second_worst', 'side_align_worst',
            'neighbor_center_x_dist', 'neighbor_center_y_dist', 'neighbor_gap', 'neighbor_center_align_best', 'neighbor_center_align_worst',
            'neighbor_side_align_best', 'neighbor_side_align_second_best', 'neighbor_side_align_second_worst', 'neighbor_side_align_worst']:

        stat_x_val = stat_x / (stat_x.max() - stat_x.min()) # normalized so max. distance between x values is 1
        if stat1.sum() <= 0 or stat2.sum() <= 0:
            dist += abs(stat2.sum() - stat1.sum())
        else:
            dist += sps.wasserstein_distance(
                u_values=stat_x_val, v_values=stat_x_val, u_weights=stat1, v_weights=stat2)
    else:
        raise ValueError(f'Unknown stat: {stat_name}')

    return dist

def egraph_set_stat_dists(stats1, stats2, std_type):

    dists = {}
    for stat_name in stats1.keys():
        dists[stat_name] = {
            'dist': egraph_set_single_stat_dist(stats1[stat_name], stats2[stat_name], stat_name),
            'std': None}
        if std_type == 'stat1':
            dists[stat_name]['std'] = stats1[stat_name]['std']
        elif std_type == 'stat2':
            dists[stat_name]['std'] = stats2[stat_name]['std']
        elif std_type == 'avg':
            dists[stat_name]['std'] = 0.5 * (stats1[stat_name]['std'] + stats2[stat_name]['std'])
        else:
            raise ValueError(f'Unknown standard deviation type: {std_type}')

    return dists

def save_stats(filename, stats):
    np.save(filename, stats)

def load_stats(filename):
    return np.load(filename, allow_pickle=True).item()

def vis_egraph_set_stat_dists(real_stats, fake_stats, stat_dists, label_names, filename=None):

    if isinstance(real_stats, str):
        real_stats = load_stats(filename=real_stats)
    if isinstance(fake_stats, str):
        fake_stats = load_stats(filename=fake_stats)
    if isinstance(stat_dists, str):
        stat_dists = load_stats(filename=stat_dists)

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
                title += f' (L2 distance: {stat_dists[stat_name]["dist"]:.8})'
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
                title += f' (average EM distance: {stat_dists[stat_name]["dist"]:.8})'
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
                title += f' (EM distance: {stat_dists[stat_name]["dist"]:.8})'
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

def compute_egraph_set_stats(out_filename, room_basepath, label_count, exterior_type, exclude_types):

    set_stats = None

    sample_names = get_room_sample_names(base_path=room_basepath)

    batch_size = 100
    batch_count = math.ceil(len(sample_names) / batch_size)
    for batch_idx in tqdm(range(batch_count)):
        samples_from = batch_size*batch_idx
        samples_to = min(batch_size*(batch_idx+1), len(sample_names))
        batch_sample_names = sample_names[samples_from:samples_to]

        room_types, room_bboxes, room_door_edges, _, room_idx_map, room_masks, _ = load_rooms(
            base_path=room_basepath, sample_names=batch_sample_names)

        stats = egraph_stats(
            node_type=room_types, nodes=room_bboxes, edges=room_door_edges,
            label_count=label_count, exterior_type=exterior_type, exclude_types=exclude_types)
        if set_stats is None:
            set_stats = stats
        else:
            for stat_name, stat in stats.items():
                set_stats[stat_name].extend(stat) # pylint: disable=E1136

    set_stats = egraph_set_stats(stats=set_stats, label_count=label_count)

    if out_filename is not None:
        os.makedirs(os.path.dirname(out_filename), exist_ok=True)
        save_stats(filename=out_filename, stats=set_stats)

    return set_stats

# either provide real_stat_filename and fake_stat_filename,
# or all the arguments below these two to compute the stats
def compute_egraph_set_stat_dists(out_dirname, real_stat_filename=None, fake_stat_filename=None, vis=True, label_names=None):

    if not os.path.exists(out_dirname):
        os.makedirs(out_dirname)

    real_set_stats = load_stats(filename=real_stat_filename)
    save_stats(filename=os.path.join(out_dirname, 'stats_real.npy'), stats=real_set_stats)

    fake_set_stats = load_stats(filename=fake_stat_filename)
    save_stats(filename=os.path.join(out_dirname, 'stats_fake.npy'), stats=fake_set_stats)

    sds = egraph_set_stat_dists(stats1=fake_set_stats, stats2=real_set_stats, std_type='stat2')

    # save stats distance as npy file and distance summary as text file
    save_stats(filename=os.path.join(out_dirname, 'stat_dists.npy'), stats=sds)
    save_egraph_set_stat_dists(stat_dists=sds, filename=os.path.join(out_dirname, 'stat_dists.txt'))

    # visualize stat difference
    if vis:
        vis_egraph_set_stat_dists(
            real_stats=real_set_stats, fake_stats=fake_set_stats, stat_dists=sds, label_names=label_names,
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
                f.write(f'{stat_name}: {stat_dists[stat_name]["dist"]:.6f}\n')

        f.write('\n')
        f.write('spatial single:\n')
        for stat_name in spatial_single_list:
            if stat_name in stat_dists:
                f.write(f'{stat_name}: {stat_dists[stat_name]["dist"]:.6f}\n')

        f.write('\n')
        f.write('spatial pair:\n')
        for stat_name in spatial_pair_list:
            if stat_name in stat_dists:
                f.write(f'{stat_name}: {stat_dists[stat_name]["dist"]:.6f}\n')

if __name__ == '__main__':

    from convert_boxes_to_rooms import room_type_names

    exterior_type = room_type_names.index('Exterior')
    exclude_types = [room_type_names.index('Wall')]
    compute_stats = True
    compute_stat_distances = True

    if compute_stats:

        result_sets = [
            # {'room_basepath': '../data/results/5_tuple_on_rplan_rooms/temp_0.9_doors_0.9_walls_0.9', 'out_filename': '../data/results/5_tuple_on_rplan_stats/temp_0.9_doors_0.9_walls_0.9/stats.npy'},
            # {'room_basepath': '../data/results/5_tuple_on_rplan_rooms/temp_0.9_doors_0.9_walls_1.0', 'out_filename': '../data/results/5_tuple_on_rplan_stats/temp_0.9_doors_0.9_walls_1.0/stats.npy'},
            # {'room_basepath': '../data/results/5_tuple_on_rplan_rooms/temp_0.9_doors_1.0_walls_0.9', 'out_filename': '../data/results/5_tuple_on_rplan_stats/temp_0.9_doors_1.0_walls_0.9/stats.npy'},
            # {'room_basepath': '../data/results/5_tuple_on_rplan_rooms/temp_0.9_doors_1.0_walls_1.0', 'out_filename': '../data/results/5_tuple_on_rplan_stats/temp_0.9_doors_1.0_walls_1.0/stats.npy'},
            # {'room_basepath': '../data/results/5_tuple_on_rplan_rooms/temp_1.0_doors_0.9_walls_0.9', 'out_filename': '../data/results/5_tuple_on_rplan_stats/temp_1.0_doors_0.9_walls_0.9/stats.npy'},
            # {'room_basepath': '../data/results/5_tuple_on_rplan_rooms/temp_1.0_doors_0.9_walls_1.0', 'out_filename': '../data/results/5_tuple_on_rplan_stats/temp_1.0_doors_0.9_walls_1.0/stats.npy'},
            # {'room_basepath': '../data/results/5_tuple_on_rplan_rooms/temp_1.0_doors_1.0_walls_0.9', 'out_filename': '../data/results/5_tuple_on_rplan_stats/temp_1.0_doors_1.0_walls_0.9/stats.npy'},
            # {'room_basepath': '../data/results/5_tuple_on_rplan_rooms/temp_1.0_doors_1.0_walls_1.0', 'out_filename': '../data/results/5_tuple_on_rplan_stats/temp_1.0_doors_1.0_walls_1.0/stats.npy'},

            # {'room_basepath': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_rooms/temp_0.9_doors_0.9_walls_0.9', 'out_filename': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_stats/temp_0.9_doors_0.9_walls_0.9/stats.npy'},
            # {'room_basepath': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_rooms/temp_0.9_doors_0.9_walls_1.0', 'out_filename': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_stats/temp_0.9_doors_0.9_walls_1.0/stats.npy'},
            # {'room_basepath': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_rooms/temp_0.9_doors_1.0_walls_0.9', 'out_filename': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_stats/temp_0.9_doors_1.0_walls_0.9/stats.npy'},
            # {'room_basepath': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_rooms/temp_0.9_doors_1.0_walls_1.0', 'out_filename': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_stats/temp_0.9_doors_1.0_walls_1.0/stats.npy'},
            # {'room_basepath': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_rooms/temp_1.0_doors_0.9_walls_0.9', 'out_filename': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_stats/temp_1.0_doors_0.9_walls_0.9/stats.npy'},
            # {'room_basepath': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_rooms/temp_1.0_doors_0.9_walls_1.0', 'out_filename': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_stats/temp_1.0_doors_0.9_walls_1.0/stats.npy'},
            # {'room_basepath': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_rooms/temp_1.0_doors_1.0_walls_0.9', 'out_filename': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_stats/temp_1.0_doors_1.0_walls_0.9/stats.npy'},
            # {'room_basepath': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_rooms/temp_1.0_doors_1.0_walls_1.0', 'out_filename': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_stats/temp_1.0_doors_1.0_walls_1.0/stats.npy'},

            # {'room_basepath': '../data/results/3_tuple_on_rplan_rooms/nodes_0.9_0.9_doors_0.9_walls_0.9', 'out_filename': '../data/results/3_tuple_on_rplan_stats/nodes_0.9_0.9_doors_0.9_walls_0.9/stats.npy'},
            # {'room_basepath': '../data/results/3_tuple_on_lifull_rooms/nodes_0.9_0.9_doors_1.0_walls_1.0', 'out_filename': '../data/results/3_tuple_on_lifull_stats/nodes_0.9_0.9_doors_1.0_walls_1.0/stats.npy'},
            # # {'room_basepath': '../data/results/3_tuple_on_rplan_rooms/nodes_0.9_0.9_doors_0.9_walls_0.9_post_edges', 'out_filename': '../data/results/3_tuple_on_rplan_stats/nodes_0.9_0.9_doors_0.9_walls_0.9_post_edges/stats.npy'},

            # {'room_basepath': '../data/results/rplan_on_rplan_rooms/rplan_on_rplan', 'out_filename': '../data/results/rplan_on_rplan_stats/stats.npy'},
            # {'room_basepath': '../data/results/rplan_on_lifull_rooms/rplan_on_lifull', 'out_filename': '../data/results/rplan_on_lifull_stats/stats.npy'},

            # {'room_basepath': '../data/results/stylegan_on_rplan_rooms/stylegan_on_rplan', 'out_filename': '../data/results/stylegan_on_rplan_stats/stats.npy'},
            # {'room_basepath': '../data/results/stylegan_on_lifull_rooms/stylegan_on_lifull', 'out_filename': '../data/results/stylegan_on_lifull_stats/stats.npy'},

            # {'room_basepath': '../data/results/graph2plan_on_rplan_rooms/graph2plan_on_rplan', 'out_filename': '../data/results/graph2plan_on_rplan_stats/stats.npy'},
            # {'room_basepath': '../data/results/graph2plan_on_lifull_rooms/graph2plan_on_lifull', 'out_filename': '../data/results/graph2plan_on_lifull_stats/stats.npy'},

            # {'room_basepath': '../data/results/3_tuple_cond_on_rplan_rooms/nodes_0.9_doors_0.9_walls_0.9', 'out_filename': '../data/results/3_tuple_cond_on_rplan_stats/stats.npy'},
            # {'room_basepath': '../data/results/3_tuple_cond_on_lifull_rooms/nodes_0.9_doors_0.9_walls_0.9', 'out_filename': '../data/results/3_tuple_cond_on_lifull_stats/stats.npy'},

            # {'room_basepath': '../data/results/gt_on_rplan_rooms/gt_on_rplan', 'out_filename': '../data/results/gt_on_rplan_stats/stats.npy'},
            # {'room_basepath': '../data/results/gt_on_lifull_rooms/gt_on_lifull', 'out_filename': '../data/results/gt_on_lifull_stats/stats.npy'},

            {'room_basepath': '../data/results/housegan_on_lifull_rooms/housegan_on_lifull', 'out_filename': '../data/results/housegan_on_lifull_stats/stats.npy'},
        ]

        for rsi, result_set in enumerate(result_sets):

            room_basepath = result_set['room_basepath']
            out_filename = result_set['out_filename']

            print(f'result set [{rsi+1}/{len(result_sets)}]: {out_filename}')

            compute_egraph_set_stats(
                out_filename=out_filename, room_basepath=room_basepath, label_count=len(room_type_names), exterior_type=exterior_type, exclude_types=exclude_types)


    if compute_stat_distances:

        stat_dist_sets = [
            # {'real_stat_filename': '../data/results/gt_on_rplan_stats/stats.npy', 'fake_stat_filename': '../data/results/5_tuple_on_rplan_stats/temp_0.9_doors_0.9_walls_0.9/stats.npy'},
            # {'real_stat_filename': '../data/results/gt_on_rplan_stats/stats.npy', 'fake_stat_filename': '../data/results/5_tuple_on_rplan_stats/temp_0.9_doors_0.9_walls_1.0/stats.npy'},
            # {'real_stat_filename': '../data/results/gt_on_rplan_stats/stats.npy', 'fake_stat_filename': '../data/results/5_tuple_on_rplan_stats/temp_0.9_doors_1.0_walls_0.9/stats.npy'},
            # {'real_stat_filename': '../data/results/gt_on_rplan_stats/stats.npy', 'fake_stat_filename': '../data/results/5_tuple_on_rplan_stats/temp_0.9_doors_1.0_walls_1.0/stats.npy'},
            # {'real_stat_filename': '../data/results/gt_on_rplan_stats/stats.npy', 'fake_stat_filename': '../data/results/5_tuple_on_rplan_stats/temp_1.0_doors_0.9_walls_0.9/stats.npy'},
            # {'real_stat_filename': '../data/results/gt_on_rplan_stats/stats.npy', 'fake_stat_filename': '../data/results/5_tuple_on_rplan_stats/temp_1.0_doors_0.9_walls_1.0/stats.npy'},
            # {'real_stat_filename': '../data/results/gt_on_rplan_stats/stats.npy', 'fake_stat_filename': '../data/results/5_tuple_on_rplan_stats/temp_1.0_doors_1.0_walls_0.9/stats.npy'},
            # {'real_stat_filename': '../data/results/gt_on_rplan_stats/stats.npy', 'fake_stat_filename': '../data/results/5_tuple_on_rplan_stats/temp_1.0_doors_1.0_walls_1.0/stats.npy'},

            # {'real_stat_filename': '../data/results/gt_on_lifull_stats/stats.npy', 'fake_stat_filename': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_stats/temp_0.9_doors_0.9_walls_0.9/stats.npy'},
            # {'real_stat_filename': '../data/results/gt_on_lifull_stats/stats.npy', 'fake_stat_filename': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_stats/temp_0.9_doors_0.9_walls_1.0/stats.npy'},
            # {'real_stat_filename': '../data/results/gt_on_lifull_stats/stats.npy', 'fake_stat_filename': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_stats/temp_0.9_doors_1.0_walls_0.9/stats.npy'},
            # {'real_stat_filename': '../data/results/gt_on_lifull_stats/stats.npy', 'fake_stat_filename': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_stats/temp_0.9_doors_1.0_walls_1.0/stats.npy'},
            # {'real_stat_filename': '../data/results/gt_on_lifull_stats/stats.npy', 'fake_stat_filename': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_stats/temp_1.0_doors_0.9_walls_0.9/stats.npy'},
            # {'real_stat_filename': '../data/results/gt_on_lifull_stats/stats.npy', 'fake_stat_filename': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_stats/temp_1.0_doors_0.9_walls_1.0/stats.npy'},
            # {'real_stat_filename': '../data/results/gt_on_lifull_stats/stats.npy', 'fake_stat_filename': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_stats/temp_1.0_doors_1.0_walls_0.9/stats.npy'},
            # {'real_stat_filename': '../data/results/gt_on_lifull_stats/stats.npy', 'fake_stat_filename': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_stats/temp_1.0_doors_1.0_walls_1.0/stats.npy'},

            # {'real_stat_filename': '../data/results/gt_on_rplan_stats/stats.npy', 'fake_stat_filename': '../data/results/3_tuple_on_rplan_stats/nodes_0.9_0.9_doors_0.9_walls_0.9/stats.npy'},
            # {'real_stat_filename': '../data/results/gt_on_lifull_stats/stats.npy', 'fake_stat_filename': '../data/results/3_tuple_on_lifull_stats/nodes_0.9_0.9_doors_1.0_walls_1.0/stats.npy'},
            # # {'real_stat_filename': '../data/results/gt_on_rplan_stats/stats.npy', 'fake_stat_filename': '../data/results/3_tuple_on_rplan_stats/nodes_0.9_0.9_doors_0.9_walls_0.9_post_edges/stats.npy'},

            # {'real_stat_filename': '../data/results/gt_on_rplan_stats/stats.npy', 'fake_stat_filename': '../data/results/rplan_on_rplan_stats/stats.npy'},
            # {'real_stat_filename': '../data/results/gt_on_lifull_stats/stats.npy', 'fake_stat_filename': '../data/results/rplan_on_lifull_stats/stats.npy'},

            # {'real_stat_filename': '../data/results/gt_on_rplan_stats/stats.npy', 'fake_stat_filename': '../data/results/stylegan_on_rplan_stats/stats.npy'},
            # {'real_stat_filename': '../data/results/gt_on_lifull_stats/stats.npy', 'fake_stat_filename': '../data/results/stylegan_on_lifull_stats/stats.npy'},

            # {'real_stat_filename': '../data/results/gt_on_rplan_stats/stats.npy', 'fake_stat_filename': '../data/results/graph2plan_on_rplan_stats/stats.npy'},
            # {'real_stat_filename': '../data/results/gt_on_lifull_stats/stats.npy', 'fake_stat_filename': '../data/results/graph2plan_on_lifull_stats/stats.npy'},

            # {'real_stat_filename': '../data/results/gt_on_rplan_stats/stats.npy', 'fake_stat_filename': '../data/results/3_tuple_cond_on_rplan_stats/stats.npy'},
            # {'real_stat_filename': '../data/results/gt_on_lifull_stats/stats.npy', 'fake_stat_filename': '../data/results/3_tuple_cond_on_lifull_stats/stats.npy'},

            {'real_stat_filename': '../data/results/gt_on_lifull_stats/stats.npy', 'fake_stat_filename': '../data/results/housegan_on_lifull_stats/stats.npy'},
        ]

        for rsi, stat_dist_set in enumerate(stat_dist_sets):
            real_stat_filename = stat_dist_set['real_stat_filename']
            fake_stat_filename = stat_dist_set['fake_stat_filename']
            out_dirname = os.path.join(os.path.dirname(fake_stat_filename), 'stat_dists')

            print(f'stat distance set [{rsi+1}/{len(stat_dist_sets)}]: {out_dirname}')

            compute_egraph_set_stat_dists(
                out_dirname=out_dirname, real_stat_filename=real_stat_filename, fake_stat_filename=fake_stat_filename, vis=True, label_names=room_type_names)
