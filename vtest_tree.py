from utils import make_rgb_indices, rplan_map, show_with_grid, make_door_indices
from glob import glob
from PIL import Image
import numpy as np
from natsort import natsorted
import matplotlib.pyplot as plt

from node import SplittingTree

from scipy.ndimage import binary_hit_or_miss
import sys
from  os.path import basename as bname
from tqdm import tqdm
import json


if __name__ == '__main__':
    print('STNODE tests')
    # errors = {}
    # img_np = np.zeros((64, 64, 3), dtype=np.uint8)
    # img_idx = np.zeros((64, 64))
    # for jj in tqdm(range(316)):
    #     IMG_PATH = f'/mnt/iscratch/datasets/rplan_ddg_var/{jj}/'

    #     IMAGES = natsorted(glob(IMG_PATH + '*_nodoor.png'))

    #     bads = []
    #     for idx in tqdm(range(len(IMAGES)), leave=False):
    #         with open(IMAGES[idx], 'rb') as fd:
    #             img_pil = Image.open(fd)
    #             img_np = np.asarray(img_pil)
    #             # img_pil.close()
    #             img_idx = make_rgb_indices(img_np, rplan_map)
                

    #         walls = img_idx == 1

    #         structure1 = np.array([[0, 1], [1, 0]])

    #         wall_corners = binary_hit_or_miss(walls, structure1=structure1)
    #         img_idx[wall_corners] = 1

    #         structure1 = np.array([[1, 0], [0, 1]])

    #         wall_corners = binary_hit_or_miss(walls, structure1=structure1, origin1=(0, -1))
    #         img_idx[wall_corners] = 1

    #         try:
    #             st = SplittingTree(img_idx, rplan_map)
    #             st.create_tree()
    #             st._merge_small_boxes(cross_wall=False)
    #             st._merge_vert_boxes(cross_wall=False)

    #         except:
    #             bads.append(IMAGES[idx])
    #     errors[jj] = bads
    #     with open('new.json', 'w') as fp:
    #         json.dump(errors, fp, indent=4)


    

    
    # sys.exit()


    # IMG_PATH = f'/mnt/iscratch/datasets/lifull_ddg_var/00/4c/9a1eb46e36a83c365331a05143eb/'
    # IMG_PATH = "/mnt/iscratch/datasets/rplan_ddg_var/292/74931_0"/mnt/iscratch/datasets/lifull_ddg_var/09/34/9d192877f550aeb9280c55b30f81/0001_
    IMG_PATH = "/mnt/iscratch/datasets/lifull_ddg_var/09/34/9d192877f550aeb9280c55b30f81/0001"

    #
    # IMG_PATH = "/mnt/iscratch/datasets/rplan_ddg_var/143/36861_0_image_nodoor.png"
    # #This one has stggered
    # IMAGES =   "/mnt/iscratch/datasets/rplan_ddg_var/114/29307_0_image_nodoor.png"
    aa = glob(IMG_PATH + '*')


    IMAGES = IMG_PATH + '_image_nodoor.png'
    DOOR_IMAGES = IMG_PATH + '_image.png'
    # print(len(IMAGES))
    # for ii, names in enumerate(IMAGES):
    #     if '59825' in names:
    #         print(ii, names)

    # print(IMAGES[66])
    img_pil = Image.open(IMAGES)
    img_np = np.asarray(img_pil)
    img_idx = make_rgb_indices(img_np, rplan_map)

    # print(DOOR_IMAGES[66])
    door_pil = Image.open(DOOR_IMAGES)
    door_np = np.asarray(door_pil)
    door_idx = make_door_indices(door_np)
    plt.imshow(door_idx)
    plt.show(False)
    print(door_idx.shape, np.unique(door_idx))
    # print(door_idx)

    # f, ax = plt.subplots(1, 3)
    # show_with_grid(rplan_map[img_idx.astype(np.uint8)], ax[0], 64)
    # plt.show()
    walls = img_idx == 1
    #
    structure1 = np.array([[0, 1], [1, 0]])
    #
    wall_corners = binary_hit_or_miss(walls, structure1=structure1)
    img_idx[wall_corners] = 1
    #
    # # # show_with_grid(wall_corners, ax[1], 64)
    #
    structure1 = np.array([[1, 0], [0, 1]])
    #
    wall_corners = binary_hit_or_miss(walls, structure1=structure1, origin1=(0, -1))
    img_idx[wall_corners] = 1
    # # show_with_grid(wall_corners, ax[2], 64)
    # plt.show()

    # sys.exit()

    st = SplittingTree(img_idx, rplan_map, grad_from='whole', door_img=door_idx)
    # st.show_grads()
    # plt.show()
    # st.split_vert = 'own'
    # st.detect_wall = 'line'
    st.create_tree()
    st._merge_small_boxes(cross_wall=False)
    st._merge_vert_boxes(cross_wall=False)
    f, ax = st.show_boxes('merged')
    # plt.savefig(f'{idx}_no_cross_wall.png', dpi=160)
    plt.show(False)
    import networkx as nx


    # sys.exit()

    # plt.show(block=True)

    # st2 = SplittingTree(img_idx, rplan_map)
    # st2.create_tree()
    # st2._merge_small_boxes(cross_wall=True)
    # st2._merge_vert_boxes(cross_wall=True)
    # f, ax = st2.show_boxes('merged')
    # plt.savefig(f'{idx}_cross_wall.png', dpi=160)

    # plt.show(block=True)
    aa = st.find_horiz_door()
    bb = st.find_vert_door()


    print(aa.edges())
    print(bb.edges())

    horiz_adj = st.find_horiz_adj()
    vert_adj = st.find_vert_adj()
    # print(horiz_adj.nodes)
    # f, ax = st.show_horiz_graph()
    f, ax = st.show_graphs()
    plt.show()





    from node import Node, Floor, LPSolver
    from random import random as rand
    floor = Floor()

    areas = []
    for rr in st.boxes:
        # print(rr.idx)
        areas.append(rr.get_area() / (64.0*64))
        # areas.append(64 / (64.0*64))
        floor.add_room(Node.from_data(rr.idx, rand(), rand(), rand(), rand()))

    floor.add_horiz_constraints(horiz_adj.edges())
    floor.add_vert_constraints(vert_adj.edges())
    print(vert_adj.edges())

    solver = LPSolver(floor)
    # solver._read_graph()
    # solver.set_min_separation(0.01)
    solver.same_line_constraints()
    solver._add_min_area_constrains(areas)

    # f, ax = plt.subplots(1, 3, figsize=(16, 16), sharex=False, sharey=False)
    solver.solve(mode=None, iter=13)

    solver._set_floor_data()

    # boxes = st.box_artist()
    # ax[1].add_collection(boxes)
    # ax[1].set_xlim((0, 64))
    # ax[1].set_ylim((64, 0))
    # ax[1].set_visible(True)
    # ax[1].set_aspect('equal')

    # show_with_grid(rplan_map[img_idx.astype(np.uint8)], ax[0], 64)face_modelv_eps_m6_mlp_lr_m4

    ax = floor.draw(ax=None, both_labels=False)
    # for aa in floor._rooms:
    #     print(aa)
    ax.set_xlim((0, 64))
    ax.set_ylim((64, 0))
    ax.set_visible(True)
    ax.set_aspect('equal')
    plt.show()


    #
    # for niter in range(10, 11):
    #     ax.cla()
    #     solver.solve(mode=None, iter=niter)
    #
    #     solver._set_floor_data()
    #
    #
    #     aa = floor.draw(ax=ax, both_labels=False)

        # plt.savefig(f'215/aa.png', dpi=160)
        # plt.

        # solver._model.reset()

    # plt.show()
    # print(solver.widths.X)
    # print(solver.heights.X)
    # print(solver.xlocs.X)
    # print(solver.ylocs.X)



    # print(solver._model.BarIterCount)

    import networkx as nx

    # aa = nx.Graph()
    # aa.add_edges_from([(1,2), (2,1), (1, 3)])
    # f, ax = plt.subplots()
    # nx.draw(aa, ax=ax)
    # plt.show()





