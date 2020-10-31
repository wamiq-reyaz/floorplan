import os
import pickle
import numpy as np
from tqdm import tqdm

def get_sample_names(box_dir, door_dir=None, wall_dir=None, sample_list_path=None):

    names = []
    if sample_list_path is None:
        if door_dir is not None and wall_dir is not None:
            # format 1:
            # three separate folders
            # each folder has one file per floor plan, and the same floor plan has the same name in each subfolder, up to the suffix:
            # boxes: (.npz)
            # door edges: (.pkl)
            # wall edges: (.pkl)

            # for path, dirnames, filenames in os.walk(box_dir):
            #     for filename in filenames:
            #         if filename.endswith('.npz') or filename.endswith('.npy'):
            #             relpath = os.path.relpath(path, start=box_dir)
            #             names.append(os.path.join(relpath if relpath != '.' else '', filename[:-len('.npz')]))

            for filename in os.listdir(box_dir):
                if os.path.isfile(os.path.join(box_dir, filename)) and (filename.endswith('.npz') or filename.endswith('.npy')):
                    names.append(os.path.join(filename[:-len('.npz')]))
        else:
            # format 2:
            # all files are in the same folder, distinguished by suffix:
            # boxes: _xyhw.npy
            # door edges: _edgelist_v.pkl and _edgelist_h.pkl
            # wall edges: walllist_all.pkl (not starting with underscore)

            for path, _, filenames in os.walk(box_dir):
                for filename in filenames:
                    if filename.endswith('_xyhw.npy'):
                        relpath = os.path.relpath(path, start=box_dir)
                        names.append(os.path.join(relpath if relpath != '.' else '', filename[:-len('_xyhw.npy')]))
    else:
        with open(sample_list_path, 'r') as f:
            names = f.read().splitlines()
        names = [n for n in names if len(n) > 0]
    
    return names

def load_boxes(sample_names, box_dir, door_dir=None, wall_dir=None, suffix=''):

    if wall_dir is None:
        wall_dir = box_dir
    if wall_dir is None:
        wall_dir = box_dir
    
    boxes = []
    door_edges = []
    wall_edges = []
    if door_dir is not None and wall_dir is not None:
        # format 1:
        # three separate folders
        # each folder has one file per floor plan, and the same floor plan has the same name in each subfolder, up to the suffix:
        # boxes: (.npz)
        # door edges: (.pkl)
        # wall edges: (.pkl)

        for name in sample_names:
            if os.path.exists(os.path.join(box_dir, f'{name}.npz')):
                boxes.append(np.load(os.path.join(box_dir, f'{name}.npz'))['arr_0'])
            else:
                boxes.append(np.load(os.path.join(box_dir, f'{name}.npy')))
            if boxes[-1].size == 0:
                boxes[-1] = np.zeros([0, 5], dtype=np.int64)

            door_edges_filename = f'{name}.pkl'
            with open(os.path.join(door_dir, door_edges_filename), 'rb') as f:
                door_edges.append(np.array(pickle.load(f)))
            if door_edges[-1].size == 0:
                door_edges[-1] = np.zeros([0, 2], dtype=np.int64)

            wall_edges_filename = f'{name}.pkl'
            with open(os.path.join(wall_dir, door_edges_filename), 'rb') as f:
                wall_edges.append(np.array(pickle.load(f)))
            if wall_edges[-1].size == 0:
                wall_edges[-1] = np.zeros([0, 2], dtype=np.int64)

    else:
        # format 2:
        # all files are in the same folder, distinguished by suffix:
        # boxes: _xyhw.npy
        # door edges: doorlist_all.pkl
        # wall edges: walllist_all.pkl (not starting with underscore)

        for name in sample_names:

            boxes.append(np.load(os.path.join(box_dir, f'{name}{suffix}_xyhw.npy')))
            if boxes[-1].size == 0:
                boxes[-1] = np.zeros([0, 5], dtype=np.int64)

            door_edges_filename = f'{name}{suffix}_doorlist_all.pkl'
            with open(os.path.join(box_dir, door_edges_filename), 'rb') as f:
                door_edges.append(np.array(pickle.load(f)))
            if door_edges[-1].size == 0:
                door_edges[-1] = np.zeros([0, 2], dtype=np.int64)

            wall_edges_filename = f'{name}{suffix}walllist_all.pkl'
            with open(os.path.join(box_dir, wall_edges_filename), 'rb') as f:
                wall_edges.append(np.array(pickle.load(f)))
            if wall_edges[-1].size == 0:
                wall_edges[-1] = np.zeros([0, 2], dtype=np.int64)

    return boxes, door_edges, wall_edges, sample_names
