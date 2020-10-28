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
from utils import make_rgb_indices, rplan_map, make_door_indices, make_rgb_indices_rounding
import networkx as nx
import os
pjoin = os.path.join
import pickle

def parse_and_save(file_list):
    for base_file_name in file_list:
        fname = base_file_name + '_image_nodoor.png'
        # print(fname)

        with open(fname, 'rb') as fd:
            img_pil = Image.open(fd)
            img_np = np.asarray(img_pil)
            img_idx = make_rgb_indices(img_np, rplan_map)

        door_fname = base_file_name + '_image.png'
        with open(door_fname, 'rb') as fd:
            door_pil = Image.open(fd)
            door_np = np.asarray(door_pil)
            door_idx = make_door_indices(door_np)

        # sys.exit()
        walls = img_idx == 1

        structure1 = np.array([[0, 1], [1, 0]])

        wall_corners = binary_hit_or_miss(walls, structure1=structure1)
        img_idx[wall_corners] = 1

        structure1 = np.array([[1, 0], [0, 1]])

        wall_corners = binary_hit_or_miss(walls, structure1=structure1, origin1=(0, -1))
        img_idx[wall_corners] = 1

        try:
            st = SplittingTree(img_idx, rplan_map, grad_from='whole', door_img=door_idx)
            st.create_tree()
            st._merge_small_boxes(cross_wall=False)
            st._merge_vert_boxes(cross_wall=False)

            # horiz_adj = st.find_horiz_adj()
            # vert_adj = st.find_vert_adj()

            horiz_wall = st.find_horiz_wall(mode='negative')
            vert_wall = st.find_vert_wall(mode='negative')

            # horiz_door = st.find_horiz_door()
            # vert_door = st.find_vert_door()

        except Exception as e:
            # raise(e)
            # sys.exit()
            bads.append(IMAGES[idx])
            continue

        num_rooms = len(st.boxes)

        # data_array = np.zeros((num_rooms, 5), np.uint8)
        # for ii, rr in enumerate(st.boxes):
        #     data_array[ii, :] = (rr.idx, rr.xmin, rr.ymin, rr.get_width(), rr.get_height())

        # with open(base_file_name + '_xyhw.npy', 'wb') as fd:
        #     np.save(fd, data_array)
        #
        # with open(base_file_name + '_edgelist_h.pkl', 'wb') as fd:
        #     pickle.dump(horiz_adj.edges(), fd, protocol=pickle.HIGHEST_PROTOCOL)
        #
        # with open(base_file_name + '_edgelist_v.pkl', 'wb') as fd:
        #     pickle.dump(vert_adj.edges(), fd, protocol=pickle.HIGHEST_PROTOCOL)
        #
        # with open(base_file_name + '_doorlist_all.pkl', 'wb') as fd:
        #     all_doors = list(horiz_door.edges()) + list(vert_door.edges())
        #     pickle.dump(all_doors, fd, protocol=pickle.HIGHEST_PROTOCOL)

        with open(base_file_name + 'negwalllist_all.pkl', 'wb') as fd:
            all_walls = list(horiz_wall.edges()) + list(vert_wall.edges())
            pickle.dump(all_walls, fd, protocol=pickle.HIGHEST_PROTOCOL)

    print('done')

if __name__ == '__main__':
    from tqdm import trange
    from multiprocessing import Pool
    p = Pool(19)
    errors = {}
    for jj in tqdm(range(1)):
        IMG_PATH = f'/mnt/iscratch/datasets/lifull_ddg_var/'
        IMG_PATH = f'/ibex/scratch/parawr/datasets/lifull_ddg_var/'

        IMG_FILE = pjoin(IMG_PATH, 'all.txt')

        with open(IMG_FILE, 'r') as fd:
            IMAGES = fd.readlines()

        # IMAGES = natsorted(glob(IMG_PATH + '*_nodoor.png'))

        bads = []
        graphs_consistent = []
        all_files = []
        tbar = trange(len(IMAGES), desc='nothing', leave=False)
        for idx in tbar:
            tbar.set_description(f'Bads: {len(bads)}')
            tbar.refresh()
            base_file_name = pjoin(IMG_PATH, IMAGES[idx].rstrip('/\n') )
            all_files.append(base_file_name)

        n_files = len(all_files)
        files_split = []
        n_elem = 2000
        for ii in range(0, n_files, n_elem):
            end_idx = min(ii+n_elem, n_files)
            files_split.append(all_files[ii:end_idx])

        p.map(parse_and_save, files_split)
        # print('the number of splits', len(files_split))
        # print('len per split')
        # for ss in files_split:
        #     print(len(ss))
        #
        # with open('lifull.json', 'w') as fp:
        #     json.dump(errors, fp, indent=4)