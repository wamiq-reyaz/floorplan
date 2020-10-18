import os, sys
import numpy as np
import pickle
from datetime import datetime
from tqdm import tqdm
from easydict import EasyDict as ED
from natsort import natsorted
from PIL import Image
from glob import glob
from utils import make_rgb_indices, rplan_map
from node import SplittingTree
from scipy.ndimage import binary_hit_or_miss
from node import STNode, AABB, Floor
import networkx as nx

import matplotlib.pyplot as plt
from scipy.stats import chisquare, wasserstein_distance
import seaborn as sns

NUM_ROOM_TYPES = rplan_map.shape[0]


def get_gt_graphs():
    stats = ED()
    NUM_ROOM_TYPES = rplan_map.shape[0]

    stats.rooms = [[] for _ in range(NUM_ROOM_TYPES)]
    stats.hadj = []
    stats.vadj = []

    errors = {}
    for jj in tqdm(range(316)):
        IMG_PATH = f'/mnt/iscratch/datasets/rplan_ddg_var/{jj}/'

        IMAGES = natsorted(glob(IMG_PATH + '*_nodoor.png'))

        bads = []
        graphs_consistent = []
        for idx in tqdm(range(len(IMAGES)), leave=False):
            with open(IMAGES[idx], 'rb') as fd:
                img_pil = Image.open(fd)
                img_np = np.asarray(img_pil)
                img_idx = make_rgb_indices(img_np, rplan_map)

            walls = img_idx == 1

            structure1 = np.array([[0, 1], [1, 0]])

            wall_corners = binary_hit_or_miss(walls, structure1=structure1)
            img_idx[wall_corners] = 1

            structure1 = np.array([[1, 0], [0, 1]])

            wall_corners = binary_hit_or_miss(walls, structure1=structure1, origin1=(0, -1))
            img_idx[wall_corners] = 1

            try:
                st = SplittingTree(img_idx, rplan_map, grad_from='whole')
                st.create_tree()
                st._merge_small_boxes(cross_wall=False)
                st._merge_vert_boxes(cross_wall=False)

            except Exception as e:
                continue
                pass
                # raise(e)
                # bads.append(IMAGES[idx])

            horiz_adj = st.find_horiz_adj()
            vert_adj = st.find_vert_adj()

            n_nodes = len(st.boxes)

            num_rooms = [0 for _ in range(NUM_ROOM_TYPES)]

            for rr in st.boxes:
                room_type = rr.idx
                num_rooms[room_type] += 1

            for ii, nn in enumerate(num_rooms):
                stats.rooms[ii].append(nn)

            stats.hadj.append(horiz_adj)
            stats.vadj.append(vert_adj)

            # print(stats)
            # if idx == 2:
            #     sys.exit()

    stats_file = './graphs_num_orig.pkl'
    if not os.path.exists(stats_file):
        with open(stats_file, 'wb') as fd:
            pickle.dump(stats, fd, protocol=pickle.HIGHEST_PROTOCOL)

    return None


def get_5_gt_stats():
    sample_files = glob('./samples/logged_0.8/*.npz')

    rand_img = np.ones((64, 64), dtype=np.uint8)

    stats = ED()
    NUM_ROOM_TYPES = rplan_map.shape[0]

    stats.rooms = [[] for _ in range(NUM_ROOM_TYPES)]
    stats.hadj = []
    stats.vadj = []

    for ii, ss in tqdm(enumerate(sample_files)):
        st = SplittingTree(rand_img, rplan_map, grad_from='whole')
        st.boxes = []

        with open(ss, 'rb') as fd:
            # print(ss)
            nodes = np.load(fd)['arr_0']

        num_rooms = [0 for _ in range(NUM_ROOM_TYPES)]

        valid = True
        for nn in nodes:
            room_id = nn[0]
            if room_id >= NUM_ROOM_TYPES:
                valid = False
                break
            st.boxes.append(STNode(AABB.from_data(nn[1],
                                                  nn[2],
                                                  nn[4],
                                                  nn[3]),
                                   idx=room_id
                                   )
                            )

            num_rooms[room_id] += 1

        if not valid:
            continue
        vadj = st.find_vert_adj()
        hadj = st.find_horiz_adj()

        for ii, nn in enumerate(num_rooms):
            stats.rooms[ii].append(nn)

        stats.hadj.append(hadj)
        stats.vadj.append(vadj)

    stats_file = './graphs_num_5.pkl'
    if not os.path.exists(stats_file):
        with open(stats_file, 'wb') as fd:
            pickle.dump(stats, fd, protocol=pickle.HIGHEST_PROTOCOL)

    return None


def get_3_stats():
    node_files = natsorted(glob('./samples/triples_0.8/*.npz'))
    hedge_files = './samples/triples_0.8/edges/h/'
    vedge_files = './samples/triples_0.8/edges/v/'

    rand_img = np.ones((64, 64), dtype=np.uint8)

    stats = ED()
    NUM_ROOM_TYPES = rplan_map.shape[0]

    stats.rooms = [[] for _ in range(NUM_ROOM_TYPES)]
    stats.hadj = []
    stats.vadj = []
    stats.idx = []

    for ii, nn in tqdm(enumerate(node_files)):
        file_base = os.path.basename(nn)
        file_base = os.path.splitext(file_base)[0]
        hh = hedge_files + file_base + '.pkl'
        vv = vedge_files + file_base + '.pkl'
        # print(vv)
        try:
            with open(nn, 'rb') as fd:
                nodes = np.load(fd)['arr_0']

            with open(hh, 'rb') as fd:
                hedge = pickle.load(fd)

            with open(vv, 'rb') as fd:
                vedge = pickle.load(fd)

        except:
            continue

        num_rooms = [0 for _ in range(NUM_ROOM_TYPES)]

        valid = True
        rooms = []
        for nn in nodes:
            room_id = nn[0]
            rooms.append(room_id)
            num_rooms[room_id] += 1
            # print('sop,etign')

        if not valid:
            continue
        vadj = vedge
        hadj = hedge

        for ii, nn in enumerate(num_rooms):
            stats.rooms[ii].append(nn)

        # print(stats.rooms)
        stats.hadj.append(hadj)
        stats.vadj.append(vadj)
        stats.idx.append(rooms)

    stats_file = './graphs_num_3.pkl'
    if not os.path.exists(stats_file):
        with open(stats_file, 'wb') as fd:
            pickle.dump(stats, fd, protocol=pickle.HIGHEST_PROTOCOL)

    return None

def convert_edges_to_idx(graph):
    nodes = graph.nodes(data=True)
    # print(nodes)
    node_to_idx = {n:data['idx'] for n, data in nodes}

    edges = graph.edges()

    idxes = []
    for e in edges:
        idxes.append(
            (node_to_idx[e[0]], node_to_idx[e[1]])
        )

    return idxes

def convert_three_edges_to_idx(edges, nodes):
    node_to_idx = {ii:nn for ii, nn in enumerate(nodes)}

    idxes = []
    for e in edges:
        idxes.append(
            (node_to_idx[e[0]], node_to_idx[e[1]])
        )

    return idxes

def get_idx_map(graph):
    nodes = graph.nodes(data=True)

    idxes = {n[0]:n[1]['idx'] for n in nodes}

    return idxes



if __name__ == '__main__':
    ####################### number DDG #################################
    with open('./graphs_num_orig.pkl', 'rb') as fd:
        orig = pickle.load(fd)

    orig_num = orig.rooms

    with open('./ddg_graph_num.pkl', 'rb') as fd:
        fiver = pickle.load(fd)

    fiver_num = fiver.rooms

    f, ax = plt.subplots(3, 4, dpi=300, figsize=(16, 16))
    f.suptitle('Num Rooms')
    floor = Floor()
    for ii in range(3):
        for jj in range(4):
            idx = ii*4 + jj
            if idx < NUM_ROOM_TYPES:
                hist_real, _, patches1 = ax[ii,jj].hist(np.array(orig_num[idx]), bins=20, range=(0,20), lw=0, alpha=0.8,
                                                  histtype='stepfilled',
                                                  label='Real', density=True)
                ax[ii,jj].hist(np.array(orig_num[idx]), bins=20, range=(0,20), lw=1, alpha=1.0, histtype='step',
                         ec=patches1[0].get_facecolor(), density=True)

                hist_sampled, _, patches2 = ax[ii,jj].hist(np.array(fiver_num[idx]), bins=20, range=(0,20), lw=0, alpha=0.8,
                                                     histtype='stepfilled',
                                                     label='Sampled 5', density=True)
                ax[ii,jj].hist(np.array(fiver_num[idx]), bins=20, range=(0,20), lw=1, alpha=1.0, histtype='step',
                         ec=patches2[0].get_facecolor(), density=True)

                emd = wasserstein_distance(hist_real, hist_sampled)
                print(ii, jj, emd)
                ax[ii, jj].text(0.5, 0.6, f'emd={emd:0.4f}', transform=ax[ii, jj].transAxes)
                ax[ii, jj].legend()
                ax[ii, jj].set_title(floor._names[idx])

    plt.savefig(f'num_rooms_ddg.png', dpi=300)
    sys.exit()

    #################### HADJ 5###########################################
    with open('./graphs_num_orig.pkl', 'rb') as fd:
        orig = pickle.load(fd)

    orig_hadj = orig.hadj

    with open('./ddg_graph_num.pkl', 'rb') as fd:
        fiver = pickle.load(fd)

    fiver_hadj = fiver.hadj

    floor = Floor()

    gt_matrix = np.zeros((NUM_ROOM_TYPES, NUM_ROOM_TYPES))
    samples_matrix = np.zeros((NUM_ROOM_TYPES, NUM_ROOM_TYPES))

    num_gt = len(orig_hadj)
    num_fiver = len(fiver_hadj)

    for hh in tqdm(orig_hadj):
        idxes = convert_edges_to_idx(hh)
        for ii, jj in idxes:
            gt_matrix[ii, jj] += 1/float(num_gt)

    for hh in tqdm(fiver_hadj):
        try:
            idxes = convert_edges_to_idx(hh)
            for ii, jj in idxes:
                samples_matrix[ii, jj] += 1 / float(num_fiver)
        except:
            print('oopsie')

    f, ax = plt.subplots(1, 2, dpi=300, figsize=(16,16))
    f.suptitle('Vert adjacencies')

    sns.heatmap(gt_matrix, ax=ax[0], xticklabels=floor._names, yticklabels=floor._names, square=True, annot=True, fmt='0.2f')
    ax[0].set_title('GT')
    sns.heatmap(samples_matrix, ax=ax[1], xticklabels=floor._names, yticklabels=floor._names, square=True, annot=True, fmt='0.2f')
    ax[1].set_title('Samples')

    plt.savefig(f'num_hadj_ddg.png', dpi=300)
    sys.exit()

    ######################## HADJ DDG######################
    with open('./graphs_num_orig.pkl', 'rb') as fd:
        orig = pickle.load(fd)

    orig_hadj = orig.hadj

    with open('./ddg_graph_num.pkl', 'rb') as fd:
        fiver = pickle.load(fd)

    fiver_hadj = fiver.vadj

    floor = Floor()

    gt_degrees = [[] for _ in range(NUM_ROOM_TYPES)]
    samples_degrees = [[] for _ in range(NUM_ROOM_TYPES)]

    num_gt = len(orig_hadj)
    num_fiver = len(fiver_hadj)

    for hh in tqdm(orig_hadj):
        idx_map = get_idx_map(hh)
        ndeg = hh.degree()

        for nn, dd in ndeg:
            curr_idx = idx_map[nn]
            gt_degrees[curr_idx].append(dd)

    for hh in tqdm(fiver_hadj):
        try:
            idx_map = get_idx_map(hh)
            ndeg = hh.degree()

            for nn, dd in ndeg:
                curr_idx = idx_map[nn]
                samples_degrees[curr_idx].append(dd)
        except:
            print('oopsie')

    f, ax = plt.subplots(3, 4, dpi=300, figsize=(16, 16))
    f.suptitle('Degrees Vert')
    floor = Floor()
    for ii in range(3):
        for jj in range(4):
            idx = ii * 4 + jj
            if idx < NUM_ROOM_TYPES:
                hist_real, _, patches1 = ax[ii, jj].hist(np.array(gt_degrees[idx]), bins=20, range=(0, 20), lw=0,
                                                         alpha=0.8,
                                                         histtype='stepfilled',
                                                         label='Real', density=True)
                ax[ii, jj].hist(np.array(gt_degrees[idx]), bins=20, range=(0, 20), lw=1, alpha=1.0, histtype='step',
                                ec=patches1[0].get_facecolor(), density=True)

                hist_sampled, _, patches2 = ax[ii, jj].hist(np.array(samples_degrees[idx]), bins=20, range=(0, 20), lw=0,
                                                            alpha=0.8,
                                                            histtype='stepfilled',
                                                            label='Sampled 5', density=True)
                ax[ii, jj].hist(np.array(samples_degrees[idx]), bins=20, range=(0, 20), lw=1, alpha=1.0, histtype='step',
                                ec=patches2[0].get_facecolor(), density=True)

                emd = wasserstein_distance(hist_real, hist_sampled)
                print(ii, jj, emd)
                ax[ii, jj].text(0.5, 0.6, f'emd={emd:0.4f}', transform=ax[ii, jj].transAxes)
                ax[ii, jj].legend()
                ax[ii, jj].set_title(floor._names[idx])

    plt.savefig(f'num_vdeg_ddg.png', dpi=300)
    sys.exit()

    ############################ HADJ 3 degree ###########################
    with open('./graphs_num_orig.pkl', 'rb') as fd:
        orig = pickle.load(fd)

    orig_hadj = orig.hadj

    with open('./graphs_num_3.pkl', 'rb') as fd:
        fiver = pickle.load(fd)


    fiver_hadj = fiver.hadj
    fiver_idx = fiver.idx

    floor = Floor()

    gt_degrees = [[] for _ in range(NUM_ROOM_TYPES)]
    samples_degrees = [[] for _ in range(NUM_ROOM_TYPES)]

    num_gt = len(orig_hadj)
    num_fiver = len(fiver_hadj)

    for hh in tqdm(orig_hadj):
        idx_map = get_idx_map(hh)
        ndeg = hh.degree()

        for nn, dd in ndeg:
            curr_idx = idx_map[nn]
            gt_degrees[curr_idx].append(dd)

    for hh, idx_ in tqdm(zip(fiver_hadj, fiver_idx)):
        try:
            graph = nx.DiGraph()
            graph.add_nodes_from([node_ for node_ in range(len(idx_))])
            graph.add_edges_from(hh)
            # print(len(idx_))
            # print(hh)
            # sys.exit()
            ndeg = graph.degree()


            for nn, dd in ndeg:
                print(hh)
                print(graph.nodes)
                print(idx_)
                curr_idx = idx_[nn]
                print(curr_idx)
                sys.exit()
                samples_degrees[curr_idx].append(dd)
        except Exception as e:
            # raise e
            print('oopsie')

    f, ax = plt.subplots(3, 4, dpi=300, figsize=(16, 16))
    f.suptitle('Degrees Horiz')
    floor = Floor()
    for ii in range(3):
        for jj in range(4):
            idx = ii * 4 + jj
            if idx < NUM_ROOM_TYPES:
                hist_real, _, patches1 = ax[ii, jj].hist(np.array(gt_degrees[idx]), bins=20, range=(0, 20), lw=0,
                                                         alpha=0.8,
                                                         histtype='stepfilled',
                                                         label='Real', density=True)
                ax[ii, jj].hist(np.array(gt_degrees[idx]), bins=20, range=(0, 20), lw=1, alpha=1.0, histtype='step',
                                ec=patches1[0].get_facecolor(), density=True)

                hist_sampled, _, patches2 = ax[ii, jj].hist(np.array(samples_degrees[idx]), bins=20, range=(0, 20),
                                                            lw=0,
                                                            alpha=0.8,
                                                            histtype='stepfilled',
                                                            label='Sampled 5', density=True)
                ax[ii, jj].hist(np.array(samples_degrees[idx]), bins=20, range=(0, 20), lw=1, alpha=1.0,
                                histtype='step',
                                ec=patches2[0].get_facecolor(), density=True)

                emd = wasserstein_distance(hist_real, hist_sampled)
                print(ii, jj, emd)
                ax[ii, jj].text(0.5, 0.6, f'emd={emd:0.4f}', transform=ax[ii, jj].transAxes)
                ax[ii, jj].legend()
                ax[ii, jj].set_title(floor._names[idx])

    plt.savefig(f'num_hdeg_3.png', dpi=300)
    sys.exit()

    ############################ HADJ 5 degree ###########################
    with open('./graphs_num_orig.pkl', 'rb') as fd:
        orig = pickle.load(fd)

    orig_hadj = orig.hadj

    with open('./graphs_num_5.pkl', 'rb') as fd:
        fiver = pickle.load(fd)

    fiver_hadj = fiver.hadj

    floor = Floor()

    gt_degrees = [[] for _ in range(NUM_ROOM_TYPES)]
    samples_degrees = [[] for _ in range(NUM_ROOM_TYPES)]

    num_gt = len(orig_hadj)
    num_fiver = len(fiver_hadj)

    for hh in tqdm(orig_hadj):
        idx_map = get_idx_map(hh)
        ndeg = hh.degree()

        for nn, dd in ndeg:
            curr_idx = idx_map[nn]
            gt_degrees[curr_idx].append(dd)

    for hh in tqdm(fiver_hadj):
        try:
            idx_map = get_idx_map(hh)
            ndeg = hh.degree()

            for nn, dd in ndeg:
                curr_idx = idx_map[nn]
                samples_degrees[curr_idx].append(dd)
        except:
            print('oopsie')

    f, ax = plt.subplots(3, 4, dpi=300, figsize=(16, 16))
    f.suptitle('Degrees Horiz')
    floor = Floor()
    for ii in range(3):
        for jj in range(4):
            idx = ii * 4 + jj
            if idx < NUM_ROOM_TYPES:
                hist_real, _, patches1 = ax[ii, jj].hist(np.array(gt_degrees[idx]), bins=20, range=(0, 20), lw=0,
                                                         alpha=0.8,
                                                         histtype='stepfilled',
                                                         label='Real', density=True)
                ax[ii, jj].hist(np.array(gt_degrees[idx]), bins=20, range=(0, 20), lw=1, alpha=1.0, histtype='step',
                                ec=patches1[0].get_facecolor(), density=True)

                hist_sampled, _, patches2 = ax[ii, jj].hist(np.array(samples_degrees[idx]), bins=20, range=(0, 20), lw=0,
                                                            alpha=0.8,
                                                            histtype='stepfilled',
                                                            label='Sampled 5', density=True)
                ax[ii, jj].hist(np.array(samples_degrees[idx]), bins=20, range=(0, 20), lw=1, alpha=1.0, histtype='step',
                                ec=patches2[0].get_facecolor(), density=True)

                emd = wasserstein_distance(hist_real, hist_sampled)
                print(ii, jj, emd)
                ax[ii, jj].text(0.5, 0.6, f'emd={emd:0.4f}', transform=ax[ii, jj].transAxes)
                ax[ii, jj].legend()
                ax[ii, jj].set_title(floor._names[idx])

    plt.savefig(f'num_hdeg_5.png', dpi=300)
    sys.exit()

    #################### HADJ 5###########################################
    # with open('./graphs_num_orig.pkl', 'rb') as fd:
    #     orig = pickle.load(fd)
    #
    # orig_hadj = orig.vadj
    #
    # with open('./graphs_num_5.pkl', 'rb') as fd:
    #     fiver = pickle.load(fd)
    #
    # fiver_hadj = fiver.vadj
    #
    # floor = Floor()
    #
    # gt_matrix = np.zeros((NUM_ROOM_TYPES, NUM_ROOM_TYPES))
    # samples_matrix = np.zeros((NUM_ROOM_TYPES, NUM_ROOM_TYPES))
    #
    # num_gt = len(orig_hadj)
    # num_fiver = len(fiver_hadj)
    #
    # for hh in tqdm(orig_hadj):
    #     idxes = convert_edges_to_idx(hh)
    #     for ii, jj in idxes:
    #         gt_matrix[ii, jj] += 1/float(num_gt)
    #
    # for hh in tqdm(fiver_hadj):
    #     try:
    #         idxes = convert_edges_to_idx(hh)
    #         for ii, jj in idxes:
    #             samples_matrix[ii, jj] += 1 / float(num_fiver)
    #     except:
    #         print('oopsie')
    #
    # f, ax = plt.subplots(1, 2, dpi=300, figsize=(16,16))
    # f.suptitle('Vert adjacencies')
    #
    # sns.heatmap(gt_matrix, ax=ax[0], xticklabels=floor._names, yticklabels=floor._names, square=True, annot=True, fmt='0.2f')
    # ax[0].set_title('GT')
    # sns.heatmap(samples_matrix, ax=ax[1], xticklabels=floor._names, yticklabels=floor._names, square=True, annot=True, fmt='0.2f')
    # ax[1].set_title('Samples')
    #
    # plt.savefig(f'num_vadj_5.png', dpi=300)
    # sys.exit()

################################################### HADJ 3#########################################
    with open('./graphs_num_orig.pkl', 'rb') as fd:
        orig = pickle.load(fd)

    orig_hadj = orig.vadj

    with open('./graphs_num_3.pkl', 'rb') as fd:
        fiver = pickle.load(fd)

    fiver_hadj = fiver.vadj
    fiver_nodes = fiver.idx

    floor = Floor()

    gt_matrix = np.zeros((NUM_ROOM_TYPES, NUM_ROOM_TYPES))
    samples_matrix = np.zeros((NUM_ROOM_TYPES, NUM_ROOM_TYPES))

    num_gt = len(orig_hadj)
    num_fiver = len(fiver_hadj)

    for hh in tqdm(orig_hadj):
        idxes = convert_edges_to_idx(hh)
        for ii, jj in idxes:
            gt_matrix[ii, jj] += 1/float(num_gt)

    for hh, nn in tqdm(zip(fiver_hadj, fiver_nodes)):
        try:
            idxes = convert_three_edges_to_idx(hh, nn)
            for ii, jj in idxes:
                samples_matrix[ii, jj] += 1 / float(num_fiver)
        except:
            print('oopsie')

    f, ax = plt.subplots(1, 2, dpi=300, figsize=(16,16))
    f.suptitle('Vert adjacencies')

    sns.heatmap(gt_matrix, ax=ax[0], xticklabels=floor._names, yticklabels=floor._names, square=True, annot=True, fmt='0.2f')
    ax[0].set_title('GT')
    sns.heatmap(samples_matrix, ax=ax[1], xticklabels=floor._names, yticklabels=floor._names, square=True, annot=True, fmt='0.2f')
    ax[1].set_title('Samples')

    plt.savefig(f'num_vadj_3.png', dpi=300)
    sys.exit()












###################################################### Graph 3 ###########################################################################

    with open('./graphs_num_orig.pkl', 'rb') as fd:
        orig = pickle.load(fd)

    orig_num = orig.rooms

    with open('./graphs_num_3.pkl', 'rb') as fd:
        fiver = pickle.load(fd)

    fiver_num = fiver.rooms

    f, ax = plt.subplots(3, 4, dpi=300, figsize=(16, 16))
    f.suptitle('Num Rooms')
    floor = Floor()
    for ii in range(3):
        for jj in range(4):
            idx = ii*4 + jj
            if idx < NUM_ROOM_TYPES:
                hist_real, _, patches1 = ax[ii,jj].hist(np.array(orig_num[idx]), bins=20, range=(0,20), lw=0, alpha=0.8,
                                                  histtype='stepfilled',
                                                  label='Real', density=True)
                ax[ii,jj].hist(np.array(orig_num[idx]), bins=20, range=(0,20), lw=1, alpha=1.0, histtype='step',
                         ec=patches1[0].get_facecolor(), density=True)

                hist_sampled, _, patches2 = ax[ii,jj].hist(np.array(fiver_num[idx]), bins=20, range=(0,20), lw=0, alpha=0.8,
                                                     histtype='stepfilled',
                                                     label='Sampled 5', density=True)
                ax[ii,jj].hist(np.array(fiver_num[idx]), bins=20, range=(0,20), lw=1, alpha=1.0, histtype='step',
                         ec=patches2[0].get_facecolor(), density=True)

                emd = wasserstein_distance(hist_real, hist_sampled)
                print(ii, jj, emd)
                ax[ii, jj].text(0.5, 0.6, f'emd={emd:0.4f}', transform=ax[ii, jj].transAxes)
                ax[ii, jj].legend()
                ax[ii, jj].set_title(floor._names[idx])

    plt.savefig(f'num_rooms_3.png', dpi=300)
    sys.exit()
    #### Grpah 5
    with open('./graphs_num_orig.pkl', 'rb') as fd:
        orig = pickle.load(fd)

    orig_num = orig.rooms

    with open('./graphs_num_5.pkl', 'rb') as fd:
        fiver = pickle.load(fd)

    fiver_num = fiver.rooms

    f, ax = plt.subplots(3, 4, dpi=300, figsize=(16, 16))
    f.suptitle('Num Rooms')
    floor = Floor()
    for ii in range(3):
        for jj in range(4):
            idx = ii*4 + jj
            if idx < NUM_ROOM_TYPES:
                hist_real, _, patches1 = ax[ii,jj].hist(np.array(orig_num[idx]), bins=20, range=(0,20), lw=0, alpha=0.8,
                                                  histtype='stepfilled',
                                                  label='Real', density=True)
                ax[ii,jj].hist(np.array(orig_num[idx]), bins=20, range=(0,20), lw=1, alpha=1.0, histtype='step',
                         ec=patches1[0].get_facecolor(), density=True)

                hist_sampled, _, patches2 = ax[ii,jj].hist(np.array(fiver_num[idx]), bins=20, range=(0,20), lw=0, alpha=0.8,
                                                     histtype='stepfilled',
                                                     label='Sampled 5', density=True)
                ax[ii,jj].hist(np.array(fiver_num[idx]), bins=20, range=(0,20), lw=1, alpha=1.0, histtype='step',
                         ec=patches2[0].get_facecolor(), density=True)

                emd = wasserstein_distance(hist_real, hist_sampled)
                print(ii, jj, emd)
                ax[ii, jj].text(0.5, 0.6, f'emd={emd:0.4f}', transform=ax[ii, jj].transAxes)
                ax[ii, jj].legend()
                ax[ii, jj].set_title(floor._names[idx])

    plt.savefig(f'num_rooms_5.png', dpi=300)
