import os, sys
import numpy as np
from PIL import Image
from glob import glob
import pickle
from natsort import natsorted



def load_pickle(fname):
    with open(fname, 'rb') as fd:
        return pickle.load(fd)

if __name__ == '__main__':
    ROOT_DIR = '/home/parawr/Projects/floorplan/samples/v2_tuned_triples_0.9/'
    verts = natsorted(glob(os.path.join(ROOT_DIR, '*.npz')))

    OTHER_DIR = './samples/v2_tuned_triples_0.9'
    SAVE_DIR = os.path.join(OTHER_DIR, 'nodes_0.9_0.9')

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR, exist_ok=True)

    from node import Node, Floor, LPSolver
    from random import random as rand
    from tqdm import tqdm

    for name in tqdm(verts):

        curr_file = name #.replace('temp_0.9', 'temp_1.0')
        base_name = os.path.basename(curr_file)
        root_name = os.path.splitext(base_name)[0]

        horiz_file = os.path.join(OTHER_DIR, 'edges', 'h_0.9', root_name + '.pkl')
        vert_file = os.path.join(OTHER_DIR, 'edges', 'v_0.9', root_name + '.pkl')

        save_file = os.path.join(SAVE_DIR, root_name + '.npy')

        try:
            horiz_edges = load_pickle(horiz_file)
            vert_edges = load_pickle(vert_file)

            vertices = np.load(curr_file)['arr_0']
        except:
            continue

        floor = Floor()

        heights = []
        widths = []
        idxes = []

        num_rooms = vertices.shape[0]
        for idx in range(num_rooms):
            id = vertices[idx, 0]
            w = vertices[idx, 1]
            h = vertices[idx, 2]

            widths.append(w/64.0)
            heights.append(h/64.0)
            floor.add_room(Node.from_data(id, rand(), rand(), rand(), rand()))

        floor.add_horiz_constraints(horiz_edges)
        floor.add_vert_constraints(vert_edges)
        floor.clear_self_loops()

        solver = LPSolver(floor)
        solver._add_xloc_constraints(widths, eps=0)
        solver._add_yloc_constraints(heights, eps=0)
        solver.same_line_constraints()

        try:
            solver.solve(mode=None)
            solver._set_floor_data()
        except:
            continue

        with open(save_file, 'wb') as fd:
            np.save(fd, solver.get_floor().get_room_array())



