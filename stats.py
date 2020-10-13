import os, sys
from scipy.ndimage import binary_hit_or_miss
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from natsort import natsorted
from glob import glob
import json

from node import SplittingTree
from utils import make_rgb_indices, rplan_map
import networkx as nx
from easydict import EasyDict as ED
import pickle



if __name__ == '__main__':
    stats = ED()
    NUM_ROOM_TYPES = rplan_map.shape[0]

    stats.n_rooms = []
    stats.widths = [[] for _ in range(NUM_ROOM_TYPES)]
    stats.heights = [[] for _ in range(NUM_ROOM_TYPES)]
    stats.xlocs = [[] for _ in range(NUM_ROOM_TYPES)]
    stats.ylocs = [[] for _ in range(NUM_ROOM_TYPES)]
    stats.hadjs = [[] for _ in range(NUM_ROOM_TYPES)]
    stats.vadjs =  [[] for _ in range(NUM_ROOM_TYPES)]


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

            # horiz_adj = st.find_horiz_adj()
            # vert_adj = st.find_vert_adj()


            n_nodes = len(st.boxes)

            stats.n_rooms.append(n_nodes)

            for rr in st.boxes:
                room_type = rr.idx
                stats.widths[room_type].append(rr.get_width())
                stats.heights[room_type].append(rr.get_height())
                stats.xlocs[room_type].append(rr.aabb.getx())
                stats.ylocs[room_type].append(rr.aabb.gety())


    stats_file = './data_stats.pkl'
    if not os.path.exists(stats_file):
        with open(stats_file, 'wb') as fd:
            pickle.dump(stats, fd, protocol=pickle.HIGHEST_PROTOCOL)

    ## plot the stats for type 5 Living Room
    plt.hist(stats.widths[5], bins=20)
    plt.show()

