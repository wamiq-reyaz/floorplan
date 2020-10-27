from glob import glob
from natsort import natsorted
import matplotlib.pyplot as plt
import numpy as np
from node import Floor





if __name__ == '__main__':
    samples = glob('samples/v2_tuned_triples_0.9/nodes_0.9_0.9/*.npy')
    print(len(samples))


    aa = Floor()
    bb = aa.load_optimized_tuple(samples[1201])
    f = plt.figure(dpi=80, figsize=(8,8))

    aa.from_array(bb)
    print(bb)
    ax = aa.draw(ax=None, both_labels=False)
    # for aa in floor._rooms:
    #     print(aa)
    ax.set_xlim((0, 64))
    ax.set_ylim((64, 0))
    ax.set_visible(True)
    ax.set_aspect('equal')
    plt.show()