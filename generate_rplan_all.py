import os, sys
from tqdm import tqdm
from natsort import natsorted
from glob import glob
import numpy as np
from scipy.ndimage import binary_hit_or_miss
from PIL import Image
import pickle
from easydict import EasyDict as ED


from node import SplittingTree
from utils import make_rgb_indices, rplan_map, make_rgb_indices_rounding, make_door_indices
import matplotlib.pyplot as plt


def sort_x_then_y(arr):
    ten_x_plus_y = 10*arr[:, 1] + arr[:, 2]
    sorted_idx = np.argsort(ten_x_plus_y)

    return arr[sorted_idx]

NUM_ROOM_TYPES = rplan_map.shape[0]

def main():
    stats = ED()
    max_rooms = 0
    max_horiz_edges = 0
    max_vert_edges = 0

    max_horiz_dict = 0
    max_vert_dict = 0
    stats.rooms = [[] for _ in range(NUM_ROOM_TYPES)]
    stats.hadj = []
    stats.vadj = []

    # IMG_PATH = f'/home/parawr/Projects/floorplan/rplan_var_images_doors/'
    # IMG_PATH = f'/home/parawr/Projects/floorplan/lifull_var_images_doors/'
    IMG_PATH = f'/mnt/iscratch/datasets/rplan_var_images/'


    FILES = IMG_PATH + 'all.txt'
    DOOR_COLOR = (228, 26, 28)
    WALL_COLOR = (153, 153, 153)

    with open(FILES, "r") as f:
        lines = f.read().splitlines()


    idx_it = tqdm(range(len(lines)), leave=False)
    for idx in idx_it:
        idx_it.set_description(f'max rooms {max_rooms}')
        img_name = IMG_PATH + lines[idx]
        img_file_name = img_name + '.png'


        with open(img_file_name, 'rb') as fd:
            img_pil = Image.open(fd)
            img_np = np.array(img_pil)
            img_np[(img_np == DOOR_COLOR).all(-1)] = WALL_COLOR

            img_idx = make_rgb_indices_rounding(img_np, rplan_map)

        door_np = np.array(img_pil)
        door_idx = make_door_indices(door_np)

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
            #
            # horiz_wall = st.find_horiz_wall()
            # vert_wall = st.find_vert_wall()

            horiz_door = st.find_horiz_door()
            vert_door = st.find_vert_door()

            print(horiz_door, vert_door)

        except Exception as e:
            print(idx, img_name)
            continue

        # num_rooms = len(st.boxes)
        #
        # data_array = np.zeros((num_rooms, 5), np.uint8)
        # for ii, rr in enumerate(st.boxes):
        #     data_array[ii, :] = (rr.idx, rr.xmin, rr.ymin, rr.get_width(), rr.get_height())
        #
        # with open(img_name + '_xyhw.npy', 'wb') as fd:
        #     np.save(fd, data_array)
        #
        # with open(img_name + '_edgelist_h.pkl', 'wb') as fd:
        #     pickle.dump(horiz_adj.edges(), fd, protocol=pickle.HIGHEST_PROTOCOL)
        #
        # with open(img_name + '_edgelist_v.pkl', 'wb') as fd:
        #     pickle.dump(vert_adj.edges(), fd, protocol=pickle.HIGHEST_PROTOCOL)
        #
        # with open(img_name + '_doorlist_all.pkl', 'wb') as fd:
        #     all_doors = list(horiz_door.edges()) + list(vert_door.edges())
        #     pickle.dump(all_doors, fd, protocol=pickle.HIGHEST_PROTOCOL)
        #
        # with open(img_name + 'walllist_all.pkl', 'wb') as fd:
        #     all_walls = list(horiz_wall.edges()) + list(vert_wall.edges())
        #     pickle.dump(all_walls, fd, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()