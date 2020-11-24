import os
import h5py
import numpy as np
# from tqdm import tqdm

furn_type_names = [
    'none', # indices start at 1 for some reason
    'window',
    'door',
    'none2', # one entry is skipped here
    'counter',
    'desk',
    'table',
    'chair',
    'refridgerator',
    'toilet',
    'sofa',
    'night_stand',
    'dresser',
    'sink',
    'shelves',
    'bookshelf',
    'cabinet',
    'bed',
    'bathtub',
    'television',
    'lamp',
    'floor_mat',
    'mirror',
    'picture',
    'whiteboard',
    'curtain',
    'blinds',
    'walkable',
    'clutter'
]

def get_sample_names(base_path=None, base_dir=None, sample_list_path=None):

    if sample_list_path is not None:
        with open(sample_list_path, 'r') as f:
            sample_names = f.read().splitlines()
        sample_names = [n for n in sample_names if len(n) > 0]
        return sample_names

    if base_path is not None:
        # format 0: five hdf5 files, each containing one property of all samples
        sample_names = []
        bboxes_file = h5py.File(base_path+'_furn_bboxes.hdf5', 'r')
        sample_names = list(bboxes_file.keys())
        # types_file.visititems(lambda name, obj: sample_names.append(name) if isinstance(obj, h5py.Dataset) else None)

    else:
        # format 1: the ddg format, one set of npy files per sample
        sample_names = []
        for path, _, filenames in os.walk(base_dir):
            for filename in filenames:
                if filename.endswith('_graph.npy'):
                    relpath = os.path.relpath(path, start=base_dir)
                    sample_names.append(os.path.join(relpath if relpath != '.' else '', filename[:-len('_graph.npy')]))

    return sample_names

def load_furniture(base_path=None, base_dir=None, sample_list_path=None, sample_names=None):
    if sample_names is None:
        sample_names = get_sample_names(base_path=base_path, base_dir=base_dir, sample_list_path=sample_list_path)

    furn_bboxes = []
    furn_neighbor_edges = []
    furn_masks = []
    room_bboxes = []
    room_masks = []

    if base_path is not None:
        # standard HDF5 format

        bboxes_file = h5py.File(base_path+'_furn_bboxes.hdf5', 'r')
        if os.path.exists(base_path+'_furn_neighbor_edges.hdf5'):
            neighbors_edges_file = h5py.File(base_path+'_furn_neighbor_edges.hdf5', 'r')
        else:
            neighbors_edges_file = None
        if os.path.exists(base_path+'_furn_masks.hdf5'):
            masks_file = h5py.File(base_path+'_furn_masks.hdf5', 'r')
        else:
            masks_file = None

        for sample_name in sample_names:
            furn_bboxes.append(np.array(bboxes_file[sample_name+'/n']))
            if neighbors_edges_file is not None:
                furn_neighbor_edges.append(np.array(neighbors_edges_file[sample_name]))
            else:
                furn_neighbor_edges.append(np.zeros(shape=[0, 2], dtype=np.int64))
            if masks_file is not None:
                furn_masks.append(np.array(masks_file[sample_name+'/n']))
            else:
                furn_masks.append(np.zeros(shape=[furn_bboxes[-1].shape[0], 64, 64], dtype=np.bool))
            room_bboxes.append(np.array(bboxes_file[sample_name+'/r']))
            if masks_file is not None:
                room_masks.append(np.array(masks_file[sample_name+'/r']))
            else:
                room_masks.append(np.zeros(shape=[1, 64, 64], dtype=np.bool))
    else:
        # Tom's dataset format
        for sample_name in sample_names:
            bboxes = np.load(os.path.join(base_dir, sample_name+'_graph.npy'))
            bboxes = bboxes[:, :6]
            bboxes[:, [1,2]] += 32 # from center origin to image min corner origin
            room_bbox = bboxes[[0]]
            bboxes = bboxes[1:]

            furn_bboxes.append(bboxes)
            furn_neighbor_edges.append(np.zeros(shape=[0, 2], dtype=np.int64))
            furn_masks.append(np.zeros(shape=[bboxes.shape[0], 64, 64], dtype=np.bool))
            room_bboxes.append(room_bbox)
            room_masks.append(np.zeros(shape=[1, 64, 64], dtype=np.bool))

    return sample_names, furn_bboxes, furn_neighbor_edges, furn_masks, room_bboxes, room_masks

def save_furniture(base_path, sample_names, furn_bboxes, furn_neighbor_edges, furn_masks, room_bboxes, room_masks, append=False):

    if any([len(x) != len(sample_names) for x in [furn_bboxes, furn_neighbor_edges, furn_masks, room_bboxes, room_masks]]):
        raise ValueError('Sample counts do not match.')

    os.makedirs(os.path.dirname(base_path), exist_ok=True)

    bboxes_file = h5py.File(base_path+'_furn_bboxes.hdf5', 'r+' if append else 'w')
    neighbors_edges_file = h5py.File(base_path+'_furn_neighbor_edges.hdf5', 'r+' if append else 'w')
    masks_file = h5py.File(base_path+'_furn_masks.hdf5', 'r+' if append else 'w')

    for i, sample_name in enumerate(sample_names):
        bboxes_file.create_dataset(sample_name+'/n', data=furn_bboxes[i])
        neighbors_edges_file.create_dataset(sample_name, data=furn_neighbor_edges[i])
        masks_file.create_dataset(sample_name+'/n', data=furn_masks[i])
        bboxes_file.create_dataset(sample_name+'/r', data=room_bboxes[i])
        masks_file.create_dataset(sample_name+'/r', data=room_masks[i])

if __name__ == '__main__':

    # # convert Tom's dataset to HDF5 format:
    # input_dir = '/home/guerrero/scratch_space/floorplan/furniture_017'
    # output_basepath = '/home/guerrero/scratch_space/floorplan/gt_furn/gt'
    # sample_names, furn_bboxes, furn_neighbor_edges, furn_masks, room_bboxes, room_masks = load_furniture(base_dir=input_dir)
    # os.makedirs(os.path.dirname(output_basepath), exist_ok=True)
    # save_furniture(
    #     base_path=output_basepath,
    #     sample_names=sample_names,
    #     furn_bboxes=furn_bboxes,
    #     furn_neighbor_edges=furn_neighbor_edges,
    #     furn_masks=furn_masks,
    #     room_bboxes=room_bboxes,
    #     room_masks=room_masks,
    #     append=False)

    # # fix labels to account for index gap at index 3 (there is no index 3)
    # # input_basepath = '../data/results/furniture/stylegan_furn/stylegan'
    # # output_basepath = '../data/results/furniture/stylegan_furn/stylegan'
    # input_basepath = '../data/results/furniture/stylegan_rnngraph/stylegan'
    # output_basepath = '../data/results/furniture/stylegan_rnngraph/stylegan'
    # sample_names, furn_bboxes, furn_neighbor_edges, furn_masks, room_bboxes, room_masks = load_furniture(base_path=input_basepath)
    # for i in range(len(furn_bboxes)):
    #     furn_bboxes[i][furn_bboxes[i][:, 0]>=3, 0] += 1
    # save_furniture(
    #     base_path=output_basepath,
    #     sample_names=sample_names,
    #     furn_bboxes=furn_bboxes,
    #     furn_neighbor_edges=furn_neighbor_edges,
    #     furn_masks=furn_masks,
    #     room_bboxes=room_bboxes,
    #     room_masks=room_masks,
    #     append=False)

    # # fix hdf5 file from wamiq
    # from tqdm import tqdm
    # input_filename = '../data/results/furniture/6_tuple_furn/furn_suppl_all.hdf5'
    # output_filename = '../data/results/furniture/6_tuple_furn/6_tuple_furn_bboxes.hdf5'
    # input_file = h5py.File(input_filename, 'r')
    # output_file = h5py.File(output_filename, 'w')
    # sample_names = list(input_file['floorplan_suppl_all'].keys())
    # for sample_name in tqdm(sample_names):
    #     output_file.create_dataset(sample_name+'/n', data=np.array(input_file['floorplan_suppl_all/'+sample_name+'/n']))
    #     output_file.create_dataset(sample_name+'/r', data=np.array(input_file['floorplan_suppl_all/'+sample_name+'/r']))

    # # convert stylegan furniture to aabbs + rotation inside the aabb
    # from tqdm import tqdm
    # input_basepath = '../data/results/furniture/stylegan_furn/stylegan_old'
    # output_basepath = '../data/results/furniture/stylegan_furn/stylegan'
    # sample_names, furn_bboxes, furn_neighbor_edges, furn_masks, room_bboxes, room_masks = load_furniture(base_path=input_basepath)
    # for si, sample_name in enumerate(tqdm(sample_names)):
    #     bboxes = furn_bboxes[si]
    #     furn_count = bboxes.shape[0]
    #     orientations = bboxes[:, 4] * ((np.pi*2) / 32)
    #     rotmats = np.array([
    #         [np.cos(orientations), -np.sin(orientations)],
    #         [np.sin(orientations),  np.cos(orientations)]]).transpose(2, 0, 1)
    #     bbox_verts = np.concatenate([
    #         bboxes[:, [1,2]] + bboxes[:, [3,4]] * np.array([[-0.5, -0.5]]),
    #         bboxes[:, [1,2]] + bboxes[:, [3,4]] * np.array([[0.5, -0.5]]),
    #         bboxes[:, [1,2]] + bboxes[:, [3,4]] * np.array([[0.5, 0.5]]),
    #         bboxes[:, [1,2]] + bboxes[:, [3,4]] * np.array([[-0.5, 0.5]])
    #         ], axis=1)
    #     bbox_verts = bbox_verts.reshape(furn_count, 4, 2)
    #     for i in range(furn_count):
    #         bbox_verts[i] = np.matmul(bbox_verts[i]-bboxes[i, [1,2]], rotmats[i].transpose())+bboxes[i, [1,2]]
    #     bbox_min = bbox_verts.min(axis=1)
    #     bbox_max = bbox_verts.max(axis=1)
    #     furn_bboxes[si][:, [1, 2]] = ((bbox_min + bbox_max) * 0.5).round().astype(np.int64) # center
    #     furn_bboxes[si][:, [3, 4]] = (bbox_max - bbox_min).round().astype(np.int64) # w,h
    # save_furniture(
    #     base_path=output_basepath,
    #     sample_names=sample_names,
    #     furn_bboxes=furn_bboxes,
    #     furn_neighbor_edges=furn_neighbor_edges,
    #     furn_masks=furn_masks,
    #     room_bboxes=room_bboxes,
    #     room_masks=room_masks,
    #     append=False)

    # # convert gt from image min corner origin to room min corner origin [+room_w/2-room_x, +room_h/2-room_y]
    # from tqdm import tqdm
    # input_basepath = '/home/guerrero/scratch_space/floorplan/gt_furn/gt_old'
    # output_basepath = '/home/guerrero/scratch_space/floorplan/gt_furn/gt'
    # sample_names, furn_bboxes, furn_neighbor_edges, furn_masks, room_bboxes, room_masks = load_furniture(base_path=input_basepath)
    # for si, sample_name in enumerate(tqdm(sample_names)):
    #     furn_bboxes[si][:, [1, 2]] += (room_bboxes[si][:, [3, 4]] * 0.5 - room_bboxes[si][:, [1, 2]]).round().astype(np.int64)
    # save_furniture(
    #     base_path=output_basepath,
    #     sample_names=sample_names,
    #     furn_bboxes=furn_bboxes,
    #     furn_neighbor_edges=furn_neighbor_edges,
    #     furn_masks=furn_masks,
    #     room_bboxes=room_bboxes,
    #     room_masks=room_masks,
    #     append=False)

    # # convert stylegan from image min corner origin to room min corner origin [+room_w/2-room_x, +room_h/2-room_y]
    # from tqdm import tqdm
    # input_basepath = '../data/results/furniture/stylegan_furn/stylegan_old2'
    # output_basepath = '../data/results/furniture/stylegan_furn/stylegan'
    # sample_names, furn_bboxes, furn_neighbor_edges, furn_masks, room_bboxes, room_masks = load_furniture(base_path=input_basepath)
    # for si, sample_name in enumerate(tqdm(sample_names)):
    #     furn_bboxes[si][:, [1, 2]] += (room_bboxes[si][:, [3, 4]] * 0.5 - room_bboxes[si][:, [1, 2]]).round().astype(np.int64)
    # save_furniture(
    #     base_path=output_basepath,
    #     sample_names=sample_names,
    #     furn_bboxes=furn_bboxes,
    #     furn_neighbor_edges=furn_neighbor_edges,
    #     furn_masks=furn_masks,
    #     room_bboxes=room_bboxes,
    #     room_masks=room_masks,
    #     append=False)

    pass
