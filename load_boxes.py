import os
import pickle
import numpy as np
import scipy.ndimage
from skimage.io import imread
import h5py
from convert_boxes_to_rooms import room_type_names

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

        box_sample_names = []
        boxes_file = h5py.File(box_dir, 'r')
        boxes_file.visititems(lambda name, obj: box_sample_names.append(name) if isinstance(obj, h5py.Dataset) else None)

        door_sample_names = []
        doors_file = h5py.File(door_dir, 'r')
        doors_file.visititems(lambda name, obj: door_sample_names.append(name) if isinstance(obj, h5py.Dataset) else None)

        wall_sample_names = []
        walls_file = h5py.File(wall_dir, 'r')
        walls_file.visititems(lambda name, obj: wall_sample_names.append(name) if isinstance(obj, h5py.Dataset) else None)

        names = sorted(list(set(box_sample_names) & set(door_sample_names) & set(wall_sample_names)))

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

def convert_boxes_to_hdf5(input_dir, output_path, append=False):

    sample_names = []
    for filename in os.listdir(input_dir):
        if os.path.isfile(os.path.join(input_dir, filename)) and (filename.endswith('.npz') or filename.endswith('.npy')):
            sample_names.append(os.path.join(filename[:-len('.npz')]))

    boxes = []
    for sample_name in sample_names:
        if os.path.exists(os.path.join(input_dir, f'{sample_name}.npz')):
            boxes.append(np.load(os.path.join(input_dir, f'{sample_name}.npz'))['arr_0'])
        else:
            boxes.append(np.load(os.path.join(input_dir, f'{sample_name}.npy')))

        if boxes[-1].size == 0:
            boxes[-1] = np.zeros([0, 5], dtype=np.int64)

    boxes_file = h5py.File(output_path, 'r+' if append else 'w')
    for i, sample_name in enumerate(sample_names):
        boxes_file.create_dataset(sample_name, data=boxes[i])

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

    os.makedirs(os.path.dirname(base_path), exist_ok=True)

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

def get_room_sample_names(base_path=None, base_dir=None, sample_list_path=None):

    if sample_list_path is not None:
        with open(sample_list_path, 'r') as f:
            sample_names = f.read().splitlines()
        sample_names = [n for n in sample_names if len(n) > 0]
        return sample_names

    if base_path is not None:
        # format 0: five hdf5 files, each containing one property of all samples
        sample_names = []
        types_file = h5py.File(base_path+'_room_types.hdf5', 'r')
        types_file.visititems(lambda name, obj: sample_names.append(name) if isinstance(obj, h5py.Dataset) else None)

    else:
        # format 1: the ddg format, one set of npy files per sample
        sample_names = []
        for path, _, filenames in os.walk(base_dir):
            for filename in filenames:
                if filename.endswith('_graph.npy'):
                    relpath = os.path.relpath(path, start=base_dir)
                    sample_names.append(os.path.join(relpath if relpath != '.' else '', filename[:-len('_graph.npy')]))

    return sample_names

# bbox: xmin, ymin, w, h
def mask_bbox(mask):
    if not mask.any():
        raise ValueError('Mask is empty.')
    if mask.ndim != 2:
        raise ValueError('Can only compute bounding box for 2-dimenaional masks.')
    mask_pix_coords = np.hstack([x.reshape(-1, 1) for x in np.nonzero(mask)])
    bbmin = mask_pix_coords[:, [1, 0]].min(axis=0)
    bbmax = mask_pix_coords[:, [1, 0]].max(axis=0)
    bbsize = (bbmax - bbmin) + 1
    return np.concatenate([bbmin, bbsize])

def load_rooms(base_path=None, base_dir=None, sample_list_path=None, sample_names=None):

    if sample_names is None:
        sample_names = get_room_sample_names(base_path=base_path, base_dir=base_dir, sample_list_path=sample_list_path)

    if base_path is not None:
        # format 0: five hdf5 files, each containing one property of all samples
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
    else:
        # format 1: the ddg format, one set of npy files per sample

        labels = [
            'exterior',
            'patio',
            'Window',
            'Door',
            'Wall',
            'Kitchen',
            'Bedroom',
            'Bathroom',
            'Living_Room',
            'Office',
            'Garage',
            'Balcony',
            'Hallway',
            'Other_Room',
        ]

        label_mapping = {
            'exterior': 'Exterior',
            'patio': 'Exterior',
            'Window': 'Wall',
            'Door': 'Wall',
            'Wall': 'Wall',
            'Kitchen': 'Kitchen',
            'Bedroom': 'Bedroom',
            'Bathroom': 'Bathroom',
            'Living_Room': 'Living Room',
            'Office': 'Office',
            'Garage': 'Other Room',
            'Balcony': 'Balcony',
            'Hallway': 'Hallway',
            'Other_Room': 'Other Room',
        }

        label_colors = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.215686, 0.494118, 0.721569],
            [0.894118, 0.101961, 0.109804],
            [0.600000, 0.600000, 0.600000],
            [0.301961, 0.686275, 0.290196],
            [0.596078, 0.305882, 0.639216],
            [1.000000, 0.498039, 0.000000],
            [1.000000, 1.000000, 0.200000],
            [0.650980, 0.337255, 0.156863],
            [0.000000, 0.000000, 0.560784],
            [0.000000, 1.000000, 1.000000],
            [1.000000, 0.960784, 1.000000],
            [0.309804, 0.305882, 0.317647],
        ]

        # imsave('/home/guerrero/scratch_space/test.png', room_masks.sum(axis=-1).astype(np.uint8)*255)

        room_types = []
        room_bboxes = []
        room_door_edges = []
        room_door_regions = []
        room_idx_maps = []
        for sample_name in sample_names:
            types_bboxes_door_adj = np.load(os.path.join(base_dir, sample_name+'_graph.npy'))

            # offset = -1
            # count = 15
            # (room_types_bboxes[:, 0] * (count-(2-offset)) + (1-offset)).round().astype(np.int64)
            types = (types_bboxes_door_adj[:, 0] * (len(labels)-2) + 1).round().astype(np.int64).reshape(-1)
            types = np.array([room_type_names.index(label_mapping[labels[type]]) for type in types], dtype=np.int64)
            room_types.append(types)

            door_adj = types_bboxes_door_adj[:, 5:]
            door_adj = (door_adj + door_adj.transpose()) > 1.0 # symmetrize
            door_edges = np.hstack([x.reshape(-1, 1) for x in np.nonzero(np.triu(door_adj, k=1))])
            room_door_edges.append(door_edges)

            # load room masks, dilate them in one direction only to cover the adjacent walls in one direction, then convert to room index map
            room_masks = np.load(os.path.join(base_dir, sample_name+'_masks.npy')).transpose(1, 0, 2) > 0.5 # room masks are transposed in the _mask.npy files
            structelm = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=np.bool)
            idx_map = np.full(room_masks.shape[:2], fill_value=labels.index('exterior'))
            for ri in range(room_masks.shape[-1]):
                room_masks[:, :, ri] = scipy.ndimage.binary_dilation(input=room_masks[:, :, ri], structure=structelm)
                idx_map[room_masks[:, :, ri]] = ri
            room_idx_maps.append(idx_map)

            bboxes = np.zeros([len(types), 4], dtype=np.int64)
            for room_idx in range(len(types)):
                room_mask = idx_map == room_idx
                if not room_mask.any():
                    bboxes[room_idx, :] = 0 # no mask found for the room, use all zeros as room bounding box
                else:
                    bboxes[room_idx, :] = mask_bbox(room_mask)
            room_bboxes.append(bboxes)

            image_filename = os.path.join(base_dir, sample_name+'.png')
            if not os.path.exists(image_filename):
                image_filename = os.path.join(base_dir, sample_name+'_image.png')
            image = imread(image_filename).astype(np.float32) / 255.0
            # door_mask = ((image - door_color.view(1, 1, -1))**2).sum(axis=-1).sqrt() < 0.1
            door_mask = ((np.expand_dims(image, axis=0) - np.array(label_colors).reshape(-1, 1, 1, 3))**2).sum(axis=-1).argmin(axis=0) == labels.index('Door')

            door_regions = np.zeros([door_edges.shape[0], 4], dtype=np.int64)
            for di, door_edge in enumerate(door_edges):
                mask1 = idx_map == door_edge[0]
                mask2 = idx_map == door_edge[1]
                mask1 = scipy.ndimage.binary_dilation(input=mask1, structure=np.ones([3, 3], dtype=np.bool))
                mask2 = scipy.ndimage.binary_dilation(input=mask2, structure=np.ones([3, 3], dtype=np.bool))
                door_region_mask = mask1 & mask2 & door_mask
                if not door_region_mask.any():
                    door_region_mask = mask1 & mask2 # no door found near the wall separating the two rooms, use the whole wall as door region
                if not door_region_mask.any():
                    door_regions[di, :] = 0 # no common wall found for the two rooms, use all-zeros as door region bounding box
                else:
                    door_regions[di, :] = mask_bbox(door_region_mask)
            room_door_regions.append(door_regions)

    return room_types, room_bboxes, room_door_edges, room_door_regions, room_idx_maps, sample_names

if __name__ == '__main__':
    # # input_dir = '../../ddg/data/results/stylegan/rplan_var_002_stylegan_baseline'
    # # output_basepath = '../data/results/stylegan_on_rplan_rooms/stylegan_on_rplan'

    # input_dir = '../../ddg/data/results/stylegan/lifull_var_002_stylegan_baseline'
    # output_basepath = '../data/results/stylegan_on_lifull_rooms/stylegan_on_lifull'

    # room_types, room_bboxes, room_door_edges, room_door_regions, room_idx_maps, sample_names = load_rooms(base_dir=input_dir)
    # save_rooms(
    #     base_path=output_basepath, sample_names=sample_names,
    #     room_types=room_types, room_bboxes=room_bboxes, room_door_edges=room_door_edges, room_door_regions=room_door_regions, room_idx_maps=room_idx_maps)

    
    # convert boxes
    # convert_boxes_to_hdf5(
    #     input_dir='../data/results/3_tuple_on_rplan/temp_0.9_old/nodes_0.9_0.9',
    #     output_path='../data/results/3_tuple_on_rplan/temp_0.9_0.9/nodes_0.9_0.9.hdf5')

    
    # fix door and edge sample names in the hdf5 files
    # filename = '../data/results/3_tuple_on_rplan/temp_0.9_0.9/doors_0.9.hdf5'
    filename = '../data/results/3_tuple_on_rplan/temp_0.9_0.9/walls_0.9.hdf5'
    with h5py.File(filename, 'r') as h5file:
        sample_names = []
        h5file.visititems(lambda name, obj: sample_names.append(name) if isinstance(obj, h5py.Dataset) else None)
        samples = []
        for sample_name in sample_names:
            samples.append(np.array(h5file[sample_name]))
    
    sample_names = [os.path.basename(name) for name in sample_names]
    
    with h5py.File(filename, 'w') as h5file:
        for i, sample_name in enumerate(sample_names):
            h5file.create_dataset(sample_name, data=samples[i])
