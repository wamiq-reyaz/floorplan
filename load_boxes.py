import os
import pickle
import numpy as np
from tqdm import tqdm

def get_sample_names(input_dir, input_list=None):

    names = []
    if input_list is None:
        if os.path.isdir(os.path.join(input_dir, 'boxes')):
            # format 1:
            # three subfolders: boxes (.npz), door_edges (.pkl), wall_edges (.pkl)
            # each subfolder has one file per floor plan, and the same floor plan has the same name in each subfolder

            format = 'multi_folder'

            # for filename in os.listdir(os.path.join(input_dir, 'boxes')):
            #     if os.path.isfile(os.path.join(input_dir, 'boxes', filename)) and filename.endswith('.npz'):
            #         names.append(filename[:-len('.npz')])

            for filename in os.listdir(os.path.join(input_dir, 'door_edges')):
                if os.path.isfile(os.path.join(input_dir, 'door_edges', filename)) and filename.endswith('.pkl'):
                    names.append(filename[:-len('.pkl')])

        else:
            # format 2:
            # all files are directly in the folder with different endings:
            # boxes: _xyhw.npy
            # door edges: _edgelist_v.pkl and _edgelist_h.pkl
            # wall edges: walllist_all.pkl (not starting with underscore)

            format = 'single_folder'

            for filename in os.listdir(input_dir):
                if os.path.isfile(os.path.join(input_dir, filename)) and filename.endswith('_xyhw.npy'):

                    names.append(filename[:-len('_xyhw.npy')])
    else:
        with open(input_list, 'r') as f:
            names = f.read().splitlines()
        names = [n for n in names if len(n) > 0]
        format = 'single_folder'
    
    return names, format

def load_boxes(input_dir, sample_names, format, suffix=''):

    boxes = []
    door_edges = []
    wall_edges = []
    if format == 'multi_folder':
        # format 1:
        # three subfolders: boxes (.npz), door_edges (.pkl), wall_edges (.pkl)
        # each subfolder has one file per floor plan, and the same floor plan has the same name in each subfolder

        for name in sample_names:
            boxes.append(np.load(os.path.join(input_dir, 'boxes', f'{name}.npz'))['arr_0'])
            if boxes[-1].size == 0:
                boxes[-1] = np.zeros([0, 5], dtype=np.int64)

            door_edges_filename = f'{name}.pkl'
            with open(os.path.join(input_dir, 'door_edges', door_edges_filename), 'rb') as f:
                door_edges.append(np.array(pickle.load(f)))
            if door_edges[-1].size == 0:
                door_edges[-1] = np.zeros([0, 2], dtype=np.int64)

            wall_edges_filename = f'{name}.pkl'
            with open(os.path.join(input_dir, 'wall_edges', door_edges_filename), 'rb') as f:
                wall_edges.append(np.array(pickle.load(f)))
            if wall_edges[-1].size == 0:
                wall_edges[-1] = np.zeros([0, 2], dtype=np.int64)

    elif format == 'single_folder':
        # format 2:
        # all files are directly in the folder with different endings:
        # boxes: _xyhw.npy
        # door edges: _edgelist_v.pkl and _edgelist_h.pkl
        # wall edges: walllist_all.pkl (not starting with underscore)

        for name in sample_names:

            boxes.append(np.load(os.path.join(input_dir, f'{name}{suffix}_xyhw.npy')))
            if boxes[-1].size == 0:
                boxes[-1] = np.zeros([0, 5], dtype=np.int64)

            door_edges_filenames = [f'{name}{suffix}_edgelist_v.pkl', f'{name}{suffix}_edgelist_h.pkl']
            door_edges.append(np.zeros([0, 2], dtype=np.int64))
            for door_edges_filename in door_edges_filenames:
                with open(os.path.join(input_dir, door_edges_filename), 'rb') as f:
                    edges = np.array(pickle.load(f))
                    if edges.size > 0:
                        door_edges[-1] = np.concatenate([door_edges[-1], edges])
            if door_edges[-1].size == 0:
                door_edges[-1] = np.zeros([0, 2], dtype=np.int64)

            wall_edges_filename = f'{name}{suffix}walllist_all.pkl'
            with open(os.path.join(input_dir, wall_edges_filename), 'rb') as f:
                wall_edges.append(np.array(pickle.load(f)))
            if wall_edges[-1].size == 0:
                wall_edges[-1] = np.zeros([0, 2], dtype=np.int64)

    else:
        raise ValueError(f'Unknown format: {format}')

    return boxes, door_edges, wall_edges, sample_names
