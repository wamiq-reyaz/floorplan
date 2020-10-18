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
from utils import make_rgb_indices, rplan_map, make_rgb_indices_rounding
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

    IMG_PATH = f'./rplan_var_images_doors/'
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
            # plt.imshow(img_np)
            # plt.show()
            img_idx = make_rgb_indices_rounding(img_np, rplan_map)
            # plt.imshow(img_idx)
            # plt.show()


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
            horiz_adj = st.find_horiz_adj()
            vert_adj = st.find_vert_adj()

            # f, ax = st.show_boxes('merged')
            # plt.savefig(f'ddg_parse_mine/{idx}_no_cross_wall.png', dpi=160)
            # plt.show()
            # break

        except Exception as e:
            print(idx, img_name)
            continue
            # raise(e)

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

    stats_file = './ddg_graph_num.pkl'
    if not os.path.exists(stats_file):
        with open(stats_file, 'wb') as fd:
            pickle.dump(stats, fd, protocol=pickle.HIGHEST_PROTOCOL)



    #################SOME OLD SHITE #######################
    #     horiz_edg_file = img_name + '_edgelist_h.pkl'
    #     horiz_dict_file = img_name + '_edge_dict_h.pkl'
    #     vert_edg_file = img_name + '_edgelist_v.pkl'
    #     vert_dict_file = img_name + '_edge_dict_v.pkl'
    #
    #     with open(horiz_edg_file, 'wb') as fd:
    #         pickle.dump(horiz_adj.edges(), fd, protocol=pickle.HIGHEST_PROTOCOL)
    #         max_horiz_edges = max(max_horiz_edges, len(horiz_adj.edges()))
    #         # print(horiz_adj.edges())
    #
    #     with open(horiz_dict_file, 'wb') as fd:
    #         adj_dict = {k:list(v.keys()) for k, v in horiz_adj.adjacency()}
    #         pickle.dump(adj_dict, fd, protocol=pickle.HIGHEST_PROTOCOL)
    #         curr_len = sum([len(v) for v in adj_dict.values()]) + len(adj_dict.keys())
    #         max_horiz_dict = max(max_horiz_dict, curr_len)
    #
    #     with open(vert_edg_file, 'wb') as fd:
    #         pickle.dump(vert_adj.edges(), fd, protocol=pickle.HIGHEST_PROTOCOL)
    #         max_vert_edges = max(max_vert_edges, len(vert_adj.edges()))
    #
    #     with open(vert_dict_file, 'wb') as fd:
    #         adj_dict = {k:list(v.keys()) for k, v in vert_adj.adjacency()}
    #         pickle.dump(adj_dict, fd, protocol=pickle.HIGHEST_PROTOCOL)
    #         curr_len = sum([len(v) for v in adj_dict.values()]) + len(adj_dict.keys())
    #         max_vert_dict = max(max_vert_dict, curr_len)
    #
    #
    #
    # print(max_rooms)
    # import json
    #
    # with open('ddg_length.json', 'w') as fd:
    #     json.dump({'hedges_max': max_horiz_edges,
    #                'hdict_max': max_horiz_dict,
    #                'vedges_max': max_vert_edges,
    #                'vdict_max': max_vert_dict},
    #               fp=fd,
    #               indent=4)



if __name__ == '__main__':
    main()