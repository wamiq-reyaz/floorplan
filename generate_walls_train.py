import os, sys
from tqdm import tqdm
from natsort import natsorted
from glob import glob
import numpy as np
from scipy.ndimage import binary_hit_or_miss
from PIL import Image
import pickle

from node import SplittingTree
from utils import make_rgb_indices, rplan_map, make_door_indices


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
        DOOR_IMAGES = natsorted(glob(IMG_PATH + '*_image.png'))

        idx_it = tqdm(range(len(IMAGES)), leave=False)
        for idx in idx_it:
            idx_it.set_description(f'max rooms {max_rooms}')
            img_name = IMAGES[idx]
            door_name = DOOR_IMAGES[idx]
            img_name = img_name[:-4]
            # img_name = img_name + '_xyhw_sorted.npy'



            #Open the rgb plan
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
                st = SplittingTree(img_idx, rplan_map, grad_from='whole', door_img=None)
                st.create_tree()
                st._merge_small_boxes(cross_wall=False)
                st._merge_vert_boxes(cross_wall=False)
                horiz_door = st.find_horiz_wall()
                vert_door = st.find_vert_wall()

                # f, ax = st.show_boxes('merged')
                # plt.savefig(f'{idx}_no_cross_wall.png', dpi=160)
                # plt.show()
                # break

            except Exception as e:
                print(jj, idx, IMAGES[idx])
                continue
                # raise(e)


            horiz_door_file = img_name + 'walllist_h.pkl'
            vert_door_file = img_name + 'walllist_v.pkl'
            all_door_file = img_name + 'walllist_all.pkl'

            with open(horiz_door_file, 'wb') as fd:
                pickle.dump(horiz_door.edges(), fd, protocol=pickle.HIGHEST_PROTOCOL)
                max_horiz_edges = max(max_horiz_edges, len(horiz_door.edges()))
                # print(horiz_adj.edges())

            with open(vert_door_file, 'wb') as fd:
                pickle.dump(vert_door.edges(), fd, protocol=pickle.HIGHEST_PROTOCOL)
                max_vert_edges = max(max_vert_edges, len(vert_door.edges()))

            with open(all_door_file, 'wb') as fd:
                all_edges = list(horiz_door.edges()) + list(vert_door.edges())
                pickle.dump(all_edges, fd, protocol=pickle.HIGHEST_PROTOCOL)
                max_vert_edges = max(max_vert_edges, len(all_edges))


    print(max_rooms)
    import json

    with open('length_wall.json', 'w') as fd:
        json.dump({'hedges_max': max_horiz_edges,
                   'hdict_max': max_horiz_dict,
                   'vedges_max': max_vert_edges,
                   'vdict_max': max_vert_dict},
                  fp=fd,
                  indent=4)



if __name__ == '__main__':
    main()