import os, sys
import numpy as np
from PIL import Image
from glob import glob
import pickle



def load_pickle(fname):
    with open(fname, 'rb') as fd:
        return pickle.load(fd)

if __name__ == '__main__':
    ROOT_DIR = './samples/triples_0.8/'
    verts = glob(os.path.join(ROOT_DIR, '*.npz'))

    curr_file = verts[3333]
    base_name = os.path.basename(curr_file)
    root_name = os.path.splitext(base_name)[0]

    horiz_file = os.path.join(ROOT_DIR, 'edges', 'h', root_name + '.pkl')
    vert_file = os.path.join(ROOT_DIR, 'edges', 'v', root_name + '.pkl')


    horiz_edges = load_pickle(horiz_file)
    vert_edges = load_pickle(vert_file)

    vertices = np.load(curr_file)['arr_0']

    print(vertices)
    print(horiz_edges)
    print(vert_edges)

    from node import Node, Floor, LPSolver
    from random import random as rand

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

    solver = LPSolver(floor)
    solver._add_xloc_constraints(widths, eps=0)
    solver._add_yloc_constraints(heights, eps=0)
    solver.same_line_constraints()

    solver.solve(mode=None)
    solver._set_floor_data()

    import matplotlib.pyplot as plt

    ax = floor.draw(ax=None, both_labels=False)
    # for aa in floor._rooms:
    #     print(aa)
    ax.set_xlim((0, 64))
    ax.set_ylim((64, 0))
    ax.set_visible(True)
    ax.set_aspect('equal')
    plt.show()



