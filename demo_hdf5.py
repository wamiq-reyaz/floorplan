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

    import h5py

    save_file = h5py.File('demo.hdf5', 'w')

    ROOT_DIR = '/home/parawr/Projects/floorplan/samples/triples_0.5/'
    verts = natsorted(glob(os.path.join(ROOT_DIR, '*.npz')))

    OTHER_DIR = './samples/triples_0.5'

    group = save_file.create_group('nodes_temp_0.5')

    sub_group = group.create_group('partial_boxes')
    sub_group = group.create_group('boxes')

    # Do nothing for the 5 tuple version


    group = save_file.create_group('optimized')





    group.create_dataset(k, data=v)



