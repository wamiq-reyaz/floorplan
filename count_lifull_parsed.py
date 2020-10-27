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
from utils import make_rgb_indices, rplan_map, make_door_indices, make_rgb_indices_rounding
import networkx as nx
import os
pjoin = os.path.join
import pickle

if __name__ == '__main__':
    from tqdm import trange

    errors = {}
    for jj in tqdm(range(1)):
        IMG_PATH = f'/mnt/iscratch/datasets/lifull_ddg_var/'
        IMG_PATH = f'/ibex/scratch/parawr/datasets/lifull_ddg_var/'

        IMG_FILE = pjoin(IMG_PATH, 'all.txt')

        with open(IMG_FILE, 'r') as fd:
            IMAGES = fd.readlines()

        # IMAGES = natsorted(glob(IMG_PATH + '*_nodoor.png'))

        bads = []
        graphs_consistent = []
        all_files = []
        tbar = trange(len(IMAGES), desc='nothing', leave=False)
        longest = 0
        print(len(IMAGES))
        print(len(IMAGES))
        for idx in tbar:
            tbar.set_description(f'length : {longest}')
            # tbar.refresh()
            base_file_name = pjoin(IMG_PATH, IMAGES[idx].rstrip('/\n') )
            fname = base_file_name + 'negwalllist_all.pkl'

            if os.path.exists(fname):
            #     yes += 1

                with open(fname, 'rb') as fd:
                    walls = pickle.load(fd)

                longest = max(longest, len(walls))




        # print(yes)
        # print(yes)
