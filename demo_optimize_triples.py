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


    ROOT_DIR = '/home/parawr/Projects/floorplan/samples/v2_tuned_triples_0.9'
    verts = natsorted(glob(os.path.join(ROOT_DIR, '*.npz')))
    print(len(verts))

    curr_file = verts[3] #.replace('temp_0.9', 'temp_1.0')
    print(curr_file)
    base_name = os.path.basename(curr_file)
    root_name = os.path.splitext(base_name)[0]

    horiz_file = os.path.join('/mnt/ibex/Projects/floorplan/samples/v2_tuned_triples_0.9', 'edges', 'h_0.9', root_name + '.pkl')
    vert_file = os.path.join('/mnt/ibex/Projects/floorplan/samples/v2_tuned_triples_0.9', 'edges', 'v_0.9', root_name + '.pkl')


    horiz_edges = load_pickle(horiz_file)
    vert_edges = load_pickle(vert_file)

    import networkx as nx
    from networkx.drawing.nx_agraph import write_dot
    graph = nx.DiGraph()
    graph.add_edges_from(vert_edges)
    write_dot(graph, 'graph.dot')
    # nx.draw(graph)
    # import matplotlib.pyplot as plt
    # plt.show()

    vertices = np.load(curr_file)['arr_0']
    # print(vertices)
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

        widths.append(w)
        heights.append(h)
        floor.add_room(Node.from_data(id, w, h, 1, 1))

    import matplotlib.pyplot as plt

    ax = floor.draw(ax=None, both_labels=False)
    for aa in floor._rooms:
        print(aa)
    ax.set_xlim((0, 64))
    ax.set_ylim((64, 0))
    ax.set_visible(True)
    ax.set_aspect('equal')
    plt.show()

    floor.add_horiz_constraints(horiz_edges)
    floor.add_vert_constraints(vert_edges)
    # floor.clear_self_loops()

    print(floor.horiz_constraints.edges())
    print(floor.vert_constraints.edges())


    solver = LPSolver(floor)
    solver.maximal_boxes_constraint(True)
    solver.same_line_constraints()


    solver._add_xloc_constraints(widths, eps=0.0)
    solver._add_yloc_constraints(heights, eps=0.0)
    print(widths)
    print(heights)

    print(len(widths))
    # solver._add_width_constraints(widths, eps=0)
    # solver._add_height_constraints(heights, eps=0)

    solver.solve(mode=None)
    solver._set_floor_data()


    ax = floor.draw(ax=None, both_labels=False)
    for aa in floor._rooms:
        print(aa)
    ax.set_xlim((0, 64))
    ax.set_ylim((64, 0))
    ax.set_visible(True)
    ax.set_aspect('equal')
    plt.show()



