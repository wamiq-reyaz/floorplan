import os
import pickle
from networkx.algorithms.operators.binary import intersection
import numpy as np
import h5py
from tqdm import tqdm

def get_box_sample_names(box_dir=None, door_dir=None, wall_dir=None, sample_list_path=None):

    if sample_list_path is not None:
        with open(sample_list_path, 'r') as f:
            names = f.read().splitlines()
        names = [n for n in names if len(n) > 0]
        return names

    names = []
    if box_dir is not None and box_dir.endswith('.hdf5'):
        # format 0:
        # three hdf5 files, one for boxes, one for door edges, one for wall edges
        boxes_file = h5py.File(box_dir, 'r')
        box_sample_names = set(boxes_file.keys())

        doors_file = h5py.File(door_dir, 'r')
        door_sample_names = set(doors_file.keys())

        walls_file = h5py.File(wall_dir, 'r')
        wall_sample_names = set(walls_file.keys())

        names = sorted(list(box_sample_names & door_sample_names & wall_sample_names))

    elif all(d is not None for d in [box_dir, door_dir, wall_dir]):
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

    return names

def load_boxes(sample_names, box_dir, door_dir=None, wall_dir=None, suffix=''):

    if wall_dir is None:
        wall_dir = box_dir
    if wall_dir is None:
        wall_dir = box_dir

    boxes = []
    door_edges = []
    wall_edges = []
    if box_dir.endswith('.hdf5'):
        # format 0:
        # three hdf5 files, one for boxes, one for door edges, one for wall edges

        boxes_file = h5py.File(box_dir, 'r')
        doors_file = h5py.File(door_dir, 'r')
        walls_file = h5py.File(wall_dir, 'r')
        for sample_name in sample_names:
            boxes.append(np.array(boxes_file[sample_name]))
            door_edges.append(np.array(doors_file[sample_name]))
            wall_edges.append(np.array(walls_file[sample_name]))

            if boxes[-1].size == 0:
                boxes[-1] = np.zeros([0, 5], dtype=np.int64)
            if door_edges[-1].size == 0:
                door_edges[-1] = np.zeros([0, 2], dtype=np.int64)
            if wall_edges[-1].size == 0:
                wall_edges[-1] = np.zeros([0, 2], dtype=np.int64)

    elif door_dir is not None and wall_dir is not None:
        # format 1:
        # three separate folders
        # each folder has one file per floor plan, and the same floor plan has the same name in each subfolder, up to the suffix:
        # boxes: (.npz)
        # door edges: (.pkl)
        # wall edges: (.pkl)

        for sample_name in sample_names:
            if os.path.exists(os.path.join(box_dir, f'{sample_name}.npz')):
                boxes.append(np.load(os.path.join(box_dir, f'{sample_name}.npz'))['arr_0'])
            else:
                boxes.append(np.load(os.path.join(box_dir, f'{sample_name}.npy')))

            door_edges_filename = f'{sample_name}.pkl'
            with open(os.path.join(door_dir, door_edges_filename), 'rb') as f:
                door_edges.append(np.array(pickle.load(f)))

            wall_edges_filename = f'{sample_name}.pkl'
            with open(os.path.join(wall_dir, door_edges_filename), 'rb') as f:
                wall_edges.append(np.array(pickle.load(f)))

            if boxes[-1].size == 0:
                boxes[-1] = np.zeros([0, 5], dtype=np.int64)
            if door_edges[-1].size == 0:
                door_edges[-1] = np.zeros([0, 2], dtype=np.int64)
            if wall_edges[-1].size == 0:
                wall_edges[-1] = np.zeros([0, 2], dtype=np.int64)

    else:
        # format 2:
        # all files are in the same folder, distinguished by suffix:
        # boxes: _xyhw.npy
        # door edges: doorlist_all.pkl
        # wall edges: walllist_all.pkl (not starting with underscore)

        for sample_name in sample_names:

            boxes.append(np.load(os.path.join(box_dir, f'{sample_name}{suffix}_xyhw.npy')))

            door_edges_filename = f'{sample_name}{suffix}_doorlist_all.pkl'
            with open(os.path.join(box_dir, door_edges_filename), 'rb') as f:
                door_edges.append(np.array(pickle.load(f)))

            wall_edges_filename = f'{sample_name}{suffix}walllist_all.pkl'
            with open(os.path.join(box_dir, wall_edges_filename), 'rb') as f:
                wall_edges.append(np.array(pickle.load(f)))

            if boxes[-1].size == 0:
                boxes[-1] = np.zeros([0, 5], dtype=np.int64)
            if door_edges[-1].size == 0:
                door_edges[-1] = np.zeros([0, 2], dtype=np.int64)
            if wall_edges[-1].size == 0:
                wall_edges[-1] = np.zeros([0, 2], dtype=np.int64)

    return boxes, door_edges, wall_edges, sample_names

def save_rooms(base_path, sample_names, room_types, room_bboxes, room_door_edges, room_door_regions, room_idx_maps, append=False):

    if any([len(x) != len(sample_names) for x in [room_types, room_bboxes, room_door_edges, room_door_regions, room_idx_maps]]):
        raise ValueError('Sample counts do not match.')

    types_file = h5py.File(base_path+'_room_types.hdf5', 'r+' if append else 'w')
    bboxes_file = h5py.File(base_path+'_room_bboxes.hdf5', 'r+' if append else 'w')
    doors_file = h5py.File(base_path+'_room_door_edges.hdf5', 'r+' if append else 'w')
    door_regions_file = h5py.File(base_path+'_room_door_regions.hdf5', 'r+' if append else 'w')
    idx_map_file = h5py.File(base_path+'_room_idx_map.hdf5', 'r+' if append else 'w')

    for i, sample_name in enumerate(sample_names):
        types_file.create_dataset(sample_name, data=room_types[i])
        bboxes_file.create_dataset(sample_name, data=room_bboxes[i])
        doors_file.create_dataset(sample_name, data=room_door_edges[i])
        door_regions_file.create_dataset(sample_name, data=room_door_regions[i])
        idx_map_file.create_dataset(sample_name, data=room_idx_maps[i])

def get_room_sample_names(base_path):
    types_file = h5py.File(base_path+'_room_types.hdf5', 'r')
    sample_names = []
    types_file.visititems(lambda name, obj: sample_names.append(name) if isinstance(obj, h5py.Dataset) else None)
    return sample_names

def load_rooms(base_path, sample_names=None):

    if sample_names is None:
        sample_names = get_room_sample_names(base_path=base_path)

    types_file = h5py.File(base_path+'_room_types.hdf5', 'r')
    bboxes_file = h5py.File(base_path+'_room_bboxes.hdf5', 'r')
    idx_map_file = h5py.File(base_path+'_room_idx_map.hdf5', 'r')
    doors_file = h5py.File(base_path+'_room_door_edges.hdf5', 'r')
    door_regions_file = h5py.File(base_path+'_room_door_regions.hdf5', 'r')

    room_types, room_bboxes, room_door_edges, room_door_regions, room_idx_maps = [], [], [], [], []
    for sample_name in sample_names:
        room_types.append(np.array(types_file[sample_name]))
        room_bboxes.append(np.array(bboxes_file[sample_name]))
        room_door_edges.append(np.array(doors_file[sample_name]))
        room_door_regions.append(np.array(door_regions_file[sample_name]))
        room_idx_maps.append(np.array(idx_map_file[sample_name]))

    return room_types, room_bboxes, room_door_edges, room_door_regions, room_idx_maps
