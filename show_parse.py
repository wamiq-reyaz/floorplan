import sys
from scipy.ndimage import binary_hit_or_miss
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from natsort import natsorted
from glob import glob
import json

from node import SplittingTree
from utils import make_rgb_indices, rplan_map



if __name__ == '__main__':
    IMAGES = '/mnt/iscratch/datasets/rplan_ddg_var/0/55_0_image_nodoor.png'


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
    f, ax = st.show_boxes('all')
    plt.show(block=False)

    # horiz_adj = list(st.find_horiz_adj().edges())
    # vert_adj = list(st.find_vert_adj().edges())

    import networkx as nx
    print('horizontal connections', list(nx.weakly_connected_components(st.find_horiz_adj())))
    print('vertical connections', list(nx.weakly_connected_components(st.find_vert_adj())))

    new_graph = st.horiz_adj.copy()
    new_graph.add_edges_from(st.vert_adj.edges())
    print('whole connections', list(nx.strongly_connected_components(new_graph)))


    print(len(st.boxes))

    f, ax = st.show_graphs()
    plt.show()

    sys.exit()