import os, sys
from tqdm import tqdm
from natsort import natsorted
from glob import glob
import numpy as np
from scipy.ndimage import binary_hit_or_miss
from PIL import Image
import pickle

from node import SplittingTree
from utils import make_rgb_indices, rplan_map


def sort_x_then_y(arr):
    ten_x_plus_y = 10*arr[:, 1] + arr[:, 2]
    sorted_idx = np.argsort(ten_x_plus_y)

    return arr[sorted_idx]


def main():
    max_rooms = 0
    max_horiz_edges = 0
    max_vert_edges = 0

    max_horiz_dict = 0
    max_vert_dict = 0
    for jj in tqdm(range(316)):
        IMG_PATH = f'/mnt/iscratch/datasets/rplan_ddg_var/{jj}/'

        IMAGES = natsorted(glob(IMG_PATH + '*_nodoor.png'))

        idx_it = tqdm(range(len(IMAGES)), leave=False)
        for idx in idx_it:
            idx_it.set_description(f'max rooms {max_rooms}')
            img_name = IMAGES[idx]
            img_name = img_name[:-4]
            # img_name = img_name + '_xyhw_sorted.npy'



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
                horiz_adj = st.find_horiz_adj()
                vert_adj = st.find_vert_adj()

                f, ax = st.show_boxes('merged')
                # plt.savefig(f'{idx}_no_cross_wall.png', dpi=160)
                plt.show()
                break

            except Exception as e:
                print(jj, idx, IMAGES[idx])
                continue
                # raise(e)


            horiz_edg_file = img_name + '_edgelist_h.pkl'
            horiz_dict_file = img_name + '_edge_dict_h.pkl'
            vert_edg_file = img_name + '_edgelist_v.pkl'
            vert_dict_file = img_name + '_edge_dict_v.pkl'

            with open(horiz_edg_file, 'wb') as fd:
                pickle.dump(horiz_adj.edges(), fd, protocol=pickle.HIGHEST_PROTOCOL)
                max_horiz_edges = max(max_horiz_edges, len(horiz_adj.edges()))
                # print(horiz_adj.edges())

            with open(horiz_dict_file, 'wb') as fd:
                adj_dict = {k:list(v.keys()) for k, v in horiz_adj.adjacency()}
                pickle.dump(adj_dict, fd, protocol=pickle.HIGHEST_PROTOCOL)
                curr_len = sum([len(v) for v in adj_dict.values()]) + len(adj_dict.keys())
                max_horiz_dict = max(max_horiz_dict, curr_len)

            with open(vert_edg_file, 'wb') as fd:
                pickle.dump(vert_adj.edges(), fd, protocol=pickle.HIGHEST_PROTOCOL)
                max_vert_edges = max(max_vert_edges, len(vert_adj.edges()))

            with open(vert_dict_file, 'wb') as fd:
                adj_dict = {k:list(v.keys()) for k, v in vert_adj.adjacency()}
                pickle.dump(adj_dict, fd, protocol=pickle.HIGHEST_PROTOCOL)
                curr_len = sum([len(v) for v in adj_dict.values()]) + len(adj_dict.keys())
                max_vert_dict = max(max_vert_dict, curr_len)

            # print(horiz_adj.nodes())
            # for k,v in horiz_adj.adjacency():
            #     print(k, v)
            #
            # print(vert_adj.nodes())
            # print(vert_adj.edges())
            # for k, v in vert_adj.adjacency():
            #     print(k, v.keys())

            # num_rooms = len(st.boxes)
            # if num_rooms > max_rooms:
            #     max_rooms = num_rooms
            #
            # data_array = np.zeros((num_rooms, 5), np.uint8)
            # for ii, rr in enumerate(st.boxes):
            #     data_array[ii, :] = (rr.idx, rr.xmin, rr.ymin, rr.get_width(), rr.get_height())
            #
            # data_array_ = sort_x_then_y(data_array)
            #
            # with open(img_name, 'wb') as fd:
            #     np.save(fd, data_array)


    print(max_rooms)
    import json

    with open('length.json', 'w') as fd:
        json.dump({'hedges_max': max_horiz_edges,
                   'hdict_max': max_horiz_dict,
                   'vedges_max': max_vert_edges,
                   'vdict_max': max_vert_dict},
                  fp=fd,
                  indent=4)



if __name__ == '__main__':
    main()