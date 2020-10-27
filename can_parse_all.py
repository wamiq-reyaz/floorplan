import sys
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



if __name__ == '__main__':
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
                # raise(e)
                # sys.exit()
                bads.append(IMAGES[idx])
                continue

            horiz_adj = st.find_horiz_adj()
            vert_adj = st.find_vert_adj()


            # n_nodes = len(st.boxes)
            # print('number of nodes,', n_nodes)
            # print(f"{IMAGES[idx]}")
            # for ii in range(n_nodes):
            #     for jj in range(ii):
            #         if ii == jj:
            #             continue
            #         else:
            #             truth = False
            #
            #             # try with ii as source
            #             reachable_hii = nx.descendants(horiz_adj, ii)
            #             reachable_vii = nx.descendants(vert_adj, ii)
            #
            #             reachable_ii = reachable_hii.union(reachable_vii)
            #
            #             truth = truth or (jj in reachable_ii)
            #
            #             # try with jj as source
            #             reachable_h = nx.descendants(horiz_adj, jj)
            #             reachable_v = nx.descendants(vert_adj, jj)
            #
            #             reachable_jj = reachable_h.union(reachable_v)
            #
            #             truth = truth or (ii in reachable_jj)
            #
            #
            #             if not truth:
            #              print(ii, reachable_hii, reachable_vii)
            #              print(jj, reachable_h, reachable_v, '\n--------------')
                         # sys.exit()




        errors[jj] = bads
        with open('lifull.json', 'w') as fp:
            json.dump(errors, fp, indent=4)