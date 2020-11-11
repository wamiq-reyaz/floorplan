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


def convert_and_save(idx_list):
    idx_list = [idx_list]
    for jj in idx_list:
        IMG_PATH = f'/mnt/iscratch/datasets/rplan_ddg_var/{jj}/'

        IMAGES = natsorted(glob(IMG_PATH + '*_nodoor.png'))

        for idx in range(len(IMAGES)):
            img_name = IMAGES[idx]
            base_img = img_name[:-(len('_nodoor.png'))]
            door_name = base_img + '.png'

            img_name = img_name[:-4]

            #Open the rgb plan
            with open(IMAGES[idx], 'rb') as fd:
                img_pil = Image.open(fd)
                img_np = np.asarray(img_pil)
                img_idx = make_rgb_indices(img_np, rplan_map)

            # open the doors image
            with open(door_name, 'rb') as fd:
                door_pil = Image.open(fd)
                door_np = np.asarray(door_pil)
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
                horiz_door = st.find_horiz_door()
                vert_door = st.find_vert_door()


            except Exception as e:
                print(jj, idx, IMAGES[idx])
                continue
                # raise(e)


            horiz_door_file = img_name + '_doorlist_h2.pkl'
            vert_door_file = img_name + '_doorlist_v2.pkl'
            all_door_file = img_name + '_doorlist_all2.pkl'

            with open(horiz_door_file, 'wb') as fd:
                pickle.dump(horiz_door.edges(), fd, protocol=pickle.HIGHEST_PROTOCOL)

            with open(vert_door_file, 'wb') as fd:
                pickle.dump(vert_door.edges(), fd, protocol=pickle.HIGHEST_PROTOCOL)

            with open(all_door_file, 'wb') as fd:
                all_edges = list(horiz_door.edges()) + list(vert_door.edges())
                pickle.dump(all_edges, fd, protocol=pickle.HIGHEST_PROTOCOL)
                print(img_name, len(all_edges))


def main2():
    print('generating in parallel')
    from multiprocessing import Pool
    p = Pool(30)

    dirs_list = list(range(316))

    p.map(convert_and_save, dirs_list)

def main1():
    max_rooms = 0
    max_horiz_edges = 0
    max_vert_edges = 0

    max_horiz_dict = 0
    max_vert_dict = 0
    for jj in tqdm(range(316)):
        IMG_PATH = f'/mnt/iscratch/datasets/rplan_ddg_var/{jj}/'

        IMAGES = natsorted(glob(IMG_PATH + '*_image_nodoor.png'))

        idx_it = tqdm(range(len(IMAGES)), leave=False)
        for idx in idx_it:
            idx_it.set_description(f'max rooms {max_rooms}')
            img_name = IMAGES[idx]
            # door_name = DOOR_IMAGES[idx]
            base_img = img_name[:-(len('_image_nodoor.png'))]
            door_name = base_img + '.png'

            img_name = img_name[:-4]

            #Open the rgb plan
            with open(IMAGES[idx], 'rb') as fd:
                img_pil = Image.open(fd)
                img_np = np.asarray(img_pil)
                img_idx = make_rgb_indices(img_np, rplan_map)

            # open the doors image
            with open(door_name, 'rb') as fd:
                door_pil = Image.open(fd)
                door_np = np.asarray(door_pil)
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
                horiz_door = st.find_horiz_door()
                vert_door = st.find_vert_door()

                # f, ax = st.show_boxes('merged')
                # plt.savefig(f'{idx}_no_cross_wall.png', dpi=160)
                # plt.show()
                # break

            except Exception as e:
                print(jj, idx, IMAGES[idx])
                continue
                # raise(e)


            horiz_door_file = img_name + '_doorlist_h2.pkl'
            vert_door_file = img_name + '_doorlist_v2.pkl'
            all_door_file = img_name + '_doorlist_all2.pkl'

            with open(horiz_door_file, 'wb') as fd:
                pickle.dump(horiz_door.edges(), fd, protocol=pickle.HIGHEST_PROTOCOL)

            with open(vert_door_file, 'wb') as fd:
                pickle.dump(vert_door.edges(), fd, protocol=pickle.HIGHEST_PROTOCOL)

            with open(all_door_file, 'wb') as fd:
                all_edges = list(horiz_door.edges()) + list(vert_door.edges())
                pickle.dump(all_edges, fd, protocol=pickle.HIGHEST_PROTOCOL)
                max_vert_edges = max(max_vert_edges, len(all_edges))

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
    main2()
    # main()