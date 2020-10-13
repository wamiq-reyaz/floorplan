from utils import make_rgb_indices, rplan_map, show_with_grid
from glob import glob
from PIL import Image
import numpy as np
from natsort import natsorted
import matplotlib.pyplot as plt

from node import SplittingTree

from scipy.ndimage import binary_hit_or_miss
import sys
import os
from os.path import basename as bname
from tqdm import tqdm
import json
from gurobipy import GRB

if __name__ == '__main__':

    ## COMMON section
    IMAGES = "/mnt/iscratch/datasets/rplan_ddg_var/68/17559_0_image_nodoor.png"




    img_pil = Image.open(IMAGES)
    img_np = np.asarray(img_pil)
    img_idx = make_rgb_indices(img_np, rplan_map)

    walls = img_idx == 1
    structure1 = np.array([[0, 1], [1, 0]])

    wall_corners = binary_hit_or_miss(walls, structure1=structure1)
    img_idx[wall_corners] = 1

    structure1 = np.array([[1, 0], [0, 1]])

    wall_corners = binary_hit_or_miss(walls, structure1=structure1, origin1=(0, -1))
    img_idx[wall_corners] = 1

    st = SplittingTree(img_idx, rplan_map, grad_from='whole')
    st.create_tree()
    st._merge_small_boxes(cross_wall=False)
    st._merge_vert_boxes(cross_wall=False)

    horiz_adj = st.find_horiz_adj()
    vert_adj = st.find_vert_adj()

    ## Reconstruction section
    from node import Node, Floor, LPSolver
    from random import random as rand

    floor = Floor()

    # TODO width constraints

    # widths = []
    # heights = []
    # epses = []
    # epses_w = []
    # for rr in st.boxes:
    #     widths.append(rr.get_width() / 64.0)
    #     heights.append(rr.get_height() / 64.0)
    #     epses.append(rand()*rr.get_height()/64.0)
    #     epses_w.append(rand()*rr.get_width()/64.0)
    #
    #     floor.add_room(Node.from_data(rr.idx, rand(), rand(), rand(), rand()))
    #
    # print(epses)
    #
    # floor.add_horiz_constraints(horiz_adj.edges())
    # floor.add_vert_constraints(vert_adj.edges())
    # solver = LPSolver(floor)
    # solver.same_line_constraints()
    #
    #
    # solver._add_height_constraints(heights=heights, eps=epses)
    # solver._add_width_constraints(widths=widths, eps=epses_w)

    # TODO Randomly dropping constraints
    widths = []
    heights = []
    epses = []
    epses_w = []
    for rr in st.boxes:
        widths.append(rr.get_width() / 64.0)
        heights.append(rr.get_height() / 64.0)
        epses.append(rand()*rr.get_height()/64.0)
        epses_w.append(rand()*rr.get_width()/64.0)
        floor.add_room(Node.from_data(rr.idx, rand(), rand(), rand(), rand()))


    solver = LPSolver(floor)
    solver._model.setParam(GRB.Param.Threads, 40)

    solver.same_line_constraints()


    horiz_edges = list(horiz_adj.edges())
    vert_edges = list(vert_adj.edges())

    EDG_REM = 2
    from random import randint as rint
    deleted_horiz = []
    for ii in range(EDG_REM):
        choice = rint(0, len(horiz_edges)- 1)
        print(horiz_edges[choice])
        deleted_horiz.append(horiz_edges[choice])
        del horiz_edges[choice]

    floor.add_horiz_constraints(horiz_edges)
    floor.add_vert_constraints(vert_edges)

    solver._add_height_constraints(heights=heights, eps=epses)
    solver._add_width_constraints(widths=widths, eps=epses_w)

    ## dropping exterior boxes
    # widths = []
    # heights = []
    # for rr in st.boxes:
    #     if not rr.idx == 0:
    #         widths.append(rr.get_width() / 64.0)
    #         heights.append(rr.get_height() / 64.0)
    #     else:
    #         widths.append(0)
    #         heights.append(0)
    #
    #     floor.add_room(Node.from_data(rr.idx, rand(), rand(), rand(), rand()))
    #
    # horiz_edges = list(horiz_adj.edges())
    # vert_edges = list(vert_adj.edges())
    #
    # def remove_exterior_edge(edgelist):
    #     valid = []
    #
    #     for ee in edgelist:
    #         if 0 not in ee:
    #             valid.append(ee)
    #
    #     return valid
    #
    # floor.add_horiz_constraints(remove_exterior_edge( horiz_edges))
    # floor.add_vert_constraints(remove_exterior_edge(vert_edges))
    #
    # # floor.add_horiz_constraints(horiz_edges)
    # # floor.add_vert_constraints(vert_edges)
    #
    # solver = LPSolver(floor)
    #
    # solver._add_height_constraints(heights=heights, eps=0.0)
    # solver._add_width_constraints(widths=widths, eps=0.0)


    f, ax = plt.subplots(1, 3, dpi=160, sharex=False, sharey=False)
    f.tight_layout()

    solver._model.setParam(GRB.Param.Presolve, 0)


    solver.solve(mode=None)
    print('solution count: ', solver._model.SolCount)
    print('objective', solver._model.getObjective().getValue())

    solver._set_floor_data()

    boxes = st.box_artist()
    ax[1].add_collection(boxes)
    ax[1].set_xlim((0, 64))
    ax[1].set_ylim((64, 0))
    ax[1].set_visible(True)
    ax[1].set_aspect('equal')

    show_with_grid(rplan_map[img_idx.astype(np.uint8)], ax[0], 64)

    floor.draw(ax=ax[2], both_labels=False)
    ax[2].set_xlim((0, 64))
    ax[2].set_ylim((64, 0))
    ax[2].set_aspect('equal')
    f.tight_layout()
    plt.show(block=False)

    f, ax = st.show_graphs()

    plt.show()


    ## SAVE WIDTH/HEIGHT relaxation

    # plt.ioff()
    # f, ax = plt.subplots(1, 1, figsize=(20, 20), dpi=160)
    # BASE_NAME = 'animation'
    # folder_name = bname(IMAGES)
    # total_path = os.path.join(BASE_NAME, folder_name)
    #
    # if not os.path.exists(total_path):
    #     os.mkdir(total_path)
    #
    # for ii in range(100):
    #     solver.solve(mode=None, iter=ii)
    #     print('solution count: ', solver._model.SolCount)
    #     print('objective', solver._model.getObjective().getValue())
    #     solver._set_floor_data()
    #
    #     ax.cla()
    #     floor.draw(ax=ax, both_labels=False, text=True, text_size=10)
    #     ax.set_xlim((0, 1))
    #     ax.set_ylim((1, 0))
    #     ax.set_aspect('equal')
    #     save_name = os.path.join(total_path, f'solve_lp_{ii:03d}.png')
    #     plt.savefig(save_name, dpi=160)
    #
    #     solver._model.reset()

