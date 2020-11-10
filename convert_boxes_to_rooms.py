import math
from itertools import chain
import numpy as np
import networkx as nx

room_type_colors = np.array(
    [[0.0, 0.0, 0.0],
    [0.600000, 0.600000, 0.600000],
    [0.301961, 0.686275, 0.290196],
    [0.596078, 0.305882, 0.639216],
    [1.000000, 0.498039, 0.000000],
    [1.000000, 1.000000, 0.200000],
    [0.650980, 0.337255, 0.156863],
    [0.000000, 1.000000, 1.000000],
    [1.000000, 0.960784, 1.000000],
    [0.309804, 0.305882, 0.317647]])

room_type_names = [
    'Exterior',
    'Wall',
    'Kitchen',
    'Bedroom',
    'Bathroom',
    'Living Room',
    'Office',
    'Balcony',
    'Hallway',
    'Other Room']

class BadBoxesException(Exception):
    pass

# boxes: (type, min_x, min_y, w, h) - N_boxes x 5
# door_edges: (from_idx, to_idx) - N_box_doors x 2
# wall_edges: (from_idx, to_idx) - N_box_walls x 2
#
# room_types: (type) N_rooms
# room_bboxes: (min_x, min_y, w, h) - N_rooms x 4
# room_door_edges: (from_idx, to_idx) - N_room_doors x 2
# room_door_regions: (min_x, min_y, w, h) - N_room_doors x 4
# room_idx_map: room index for each pixel - img_res_y x img_res_x
def convert_boxes_to_rooms(boxes, door_edges, wall_edges, img_res, room_type_count):

    exterior_type_idx = room_type_names.index('Exterior')
    
    # convert all the boxes in all floor plans to rooms
    room_types, room_bboxes, room_door_edges, room_door_regions, room_idx_maps = [], [], [], [], []
    for fp_boxes, fp_door_edges, fp_wall_edges in zip(boxes, door_edges, wall_edges):

        if fp_boxes.ndim != 2 or fp_boxes.shape[1] != 5:
            raise BadBoxesException('Invalid boxes, boxes array has incorrect shape.')

        if fp_door_edges.ndim != 2 or fp_door_edges.shape[1] != 2 or fp_wall_edges.ndim != 2 or fp_wall_edges.shape[1] != 2:
            raise BadBoxesException('Invalid edges, edges array has incorrect shape.')

        if fp_boxes.size == 0:
            raise BadBoxesException('Empty boxes.')

        if (fp_door_edges.size > 0  and fp_door_edges.max() >= fp_boxes.shape[0]) or (fp_wall_edges.size > 0 and fp_wall_edges.max() >= fp_boxes.shape[0]):
            raise BadBoxesException('Invalid edges, the index is out of bounds.')

        if fp_boxes.dtype == np.float32:
            # boxes are probably 3-tuples which are stored in float format, where the max and min corners of boxes are in [0..1] instead of [0..img_res]
            # (with integer coordinates, boxes start at lower left pixel corner and end at upper right pixel corner, giving an img_res+1 x img_res+1 grid)
            fp_boxes[:, [1, 3]] *= img_res[1]
            fp_boxes[:, [2, 4]] *= img_res[0]

            fp_boxes[:, 3:] += fp_boxes[:, 1:3] # convert from w,h to max corner
            fp_boxes = fp_boxes.round().astype(np.int64)
            fp_boxes[:, 3:] -= fp_boxes[:, 1:3] # convert from max corner back to w,h

        if fp_boxes.size > 0 and (fp_boxes[:, 0].max() >= room_type_count or fp_boxes[:, 1].max() >= img_res[1] or fp_boxes[:, 2].max() >= img_res[0] or (fp_boxes[:, 1]+fp_boxes[:, 3]).max() > img_res[1] or (fp_boxes[:, 2]+fp_boxes[:, 4]).max() > img_res[0]):
            raise BadBoxesException('Invalid boxes, the type index or box shapes are out of bounds.')

        # boxes: x,y,w,h (not x,y,h,w, as the file ending suggests)

        # get adjacency matrix for door edges
        door_adj = np.zeros(shape=[fp_boxes.shape[0], fp_boxes.shape[0]], dtype=np.bool)
        door_adj[fp_door_edges[:, 0], fp_door_edges[:, 1]] = True
        door_adj = np.logical_or(door_adj, np.transpose(door_adj)) # make symmetric

        # get adjacency matrix for wall edges
        wall_adj = np.zeros(shape=[fp_boxes.shape[0], fp_boxes.shape[0]], dtype=np.bool)
        wall_adj[fp_wall_edges[:, 0], fp_wall_edges[:, 1]] = True
        wall_adj = np.logical_or(wall_adj, np.transpose(wall_adj)) # make symmetric

        # create same room edges between adjacent (or overlapping) rooms that have the same type and are not connected by wall edges
        same_room_edges = []
        for i in range(fp_boxes.shape[0]):
            for j in range(i+1, fp_boxes.shape[0]):
                if fp_boxes[i, 0] == fp_boxes[j, 0]: # same type
                    if not wall_adj[i, j]: # without separating wall
                        xoverlap = not (fp_boxes[i, 1]+fp_boxes[i, 3] <= fp_boxes[j, 1] or fp_boxes[i, 1] >= fp_boxes[j, 1]+fp_boxes[j, 3])
                        xtouch = fp_boxes[i, 1]+fp_boxes[i, 3] == fp_boxes[j, 1] or fp_boxes[i, 1] == fp_boxes[j, 1]+fp_boxes[j, 3]
                        yoverlap = not (fp_boxes[i, 2]+fp_boxes[i, 4] <= fp_boxes[j, 2] or fp_boxes[i, 2] >= fp_boxes[j, 2]+fp_boxes[j, 4])
                        ytouch = fp_boxes[i, 2]+fp_boxes[i, 4] == fp_boxes[j, 2] or fp_boxes[i, 2] == fp_boxes[j, 2]+fp_boxes[j, 4]
                        if (xoverlap and yoverlap) or (xoverlap and ytouch) or (yoverlap and xtouch): # adjacent
                            same_room_edges.append([i, j])
        same_room_edges = np.array(same_room_edges)

        # get connected components of graph with same room edges; these form the rooms
        same_room_graph = nx.Graph()
        same_room_graph.add_nodes_from(list(range(fp_boxes.shape[0])))
        same_room_graph.add_edges_from(same_room_edges.tolist())
        same_room_conn_comp = [list(comp) for comp in nx.connected_components(same_room_graph)]

        # merge all components of type 'exterior' (with type index 1) into a single component
        exterior_comp = [comp for comp in same_room_conn_comp if fp_boxes[comp[0], 0] == exterior_type_idx]
        nonexterior_comp = [comp for comp in same_room_conn_comp if fp_boxes[comp[0], 0] != exterior_type_idx]
        same_room_conn_comp = [list(chain(*exterior_comp))] + nonexterior_comp

        # create room masks, types, bounding boxes, and door edges
        fp_room_types = np.zeros([len(same_room_conn_comp)], dtype=np.int64)
        fp_room_bboxes = np.zeros([len(same_room_conn_comp), 4], dtype=np.int64)
        fp_room_door_adj = np.zeros([len(same_room_conn_comp), len(same_room_conn_comp)], dtype=np.bool)
        fp_room_idx_map = np.full([*img_res], fill_value=exterior_type_idx, dtype=np.int64)
        for comp_ind, comp in enumerate(same_room_conn_comp):

            if np.unique(fp_boxes[comp, 0]).size != 1:
                raise ValueError('Found set of boxes with mixed types that are connected by same room edges.')
                # this should not happen since I never generate same room edges between boxes of different type

            # set room type
            fp_room_types[comp_ind] = fp_boxes[comp[0], 0]

            fp_room_bboxes[comp_ind, :] = fp_boxes[comp[0], 1:]
            for box_ind in comp:
                # update room mask
                fp_room_idx_map[
                    fp_boxes[box_ind, 2]:fp_boxes[box_ind, 2]+fp_boxes[box_ind, 4],
                    fp_boxes[box_ind, 1]:fp_boxes[box_ind, 1]+fp_boxes[box_ind, 3]] = comp_ind

                # update room bounding box
                fp_room_bboxes[comp_ind, :2] = np.minimum(fp_room_bboxes[comp_ind, :2], fp_boxes[box_ind, 1:3])
                fp_room_bboxes[comp_ind, 2:] = np.maximum(fp_boxes[box_ind, 1:3]+fp_boxes[box_ind, 3:5]-fp_room_bboxes[comp_ind, :2], fp_room_bboxes[comp_ind, 2:] - fp_room_bboxes[comp_ind, :2])

            # update room door edges (room door edge probability is the maximum probability of any edge that connects any box in room 1 to any box in room 2)
            for comp_ind2, comp2 in enumerate(same_room_conn_comp):
                fp_room_door_adj[comp_ind, comp_ind2] = door_adj[comp, :][:, comp2].any()
        
        # convert room door edges from adjacency matrix to edge list
        fp_room_door_edges = np.hstack([x.reshape(-1, 1) for x in np.nonzero(np.triu(fp_room_door_adj, k=1))])

        # get door regions
        fp_room_door_regions = np.zeros([fp_room_door_edges.shape[0], 4], dtype=np.int64)
        for ei, room_door_edge in enumerate(fp_room_door_edges):
            comp1 = np.array(same_room_conn_comp[room_door_edge[0]])
            comp2 = np.array(same_room_conn_comp[room_door_edge[1]])

            door_from, door_to = np.nonzero(door_adj[comp1, :][:, comp2])
            if len(door_from) == 0:
                raise ValueError('Found door edge between two rooms that do not have a door edge between any of their boxes.')
                # this should not happen since I only generate door edges between rooms that have at least one door edge between any of their boxes

            door_from = comp1[door_from] # pick first door edge as door location
            door_to = comp2[door_to]

            door_region = None
            for i, j in zip(door_from, door_to):
                xoverlap = not (fp_boxes[i, 1]+fp_boxes[i, 3] <= fp_boxes[j, 1] or fp_boxes[i, 1] >= fp_boxes[j, 1]+fp_boxes[j, 3])
                xtouch = fp_boxes[i, 1]+fp_boxes[i, 3] == fp_boxes[j, 1] or fp_boxes[i, 1] == fp_boxes[j, 1]+fp_boxes[j, 3]
                yoverlap = not (fp_boxes[i, 2]+fp_boxes[i, 4] <= fp_boxes[j, 2] or fp_boxes[i, 2] >= fp_boxes[j, 2]+fp_boxes[j, 4])
                ytouch = fp_boxes[i, 2]+fp_boxes[i, 4] == fp_boxes[j, 2] or fp_boxes[i, 2] == fp_boxes[j, 2]+fp_boxes[j, 4]
                if (xoverlap and yoverlap) or (xoverlap and ytouch) or (yoverlap and xtouch): # adjacent
                    door_region = np.array([
                        fp_boxes[[i, j], 1].max(),
                        fp_boxes[[i, j], 2].max(),
                        (fp_boxes[[i, j], 1]+fp_boxes[[i, j], 3]).min() - fp_boxes[[i, j], 1].max(),
                        (fp_boxes[[i, j], 2]+fp_boxes[[i, j], 4]).min() - fp_boxes[[i, j], 2].max()])
                    break

            if door_region is None:
                raise BadBoxesException('Invalid boxes, there is a door edge between boxes that do not touch.')

            fp_room_door_regions[ei, :] = door_region

        room_types.append(fp_room_types)
        room_bboxes.append(fp_room_bboxes)
        room_door_edges.append(fp_room_door_edges)
        room_door_regions.append(fp_room_door_regions)
        room_idx_maps.append(fp_room_idx_map)

    return room_types, room_bboxes, room_door_edges, room_door_regions, room_idx_maps

if __name__ == '__main__':
    import os
    from load_boxes import get_box_sample_names, load_boxes, save_rooms
    from convert_boxes_to_rooms import convert_boxes_to_rooms, room_type_colors
    from tqdm import tqdm

    result_sets = [
        # {'box_dir': '../data/results/5_tuple_on_rplan/temp_0.9', 'door_dir': '../data/results/5_tuple_on_rplan/temp_0.9/doors_0.9', 'wall_dir': '../data/results/5_tuple_on_rplan/temp_0.9/walls_0.9', 'sample_list': '../data/results/5_tuple_on_rplan/temp_0.9/test_doors_0.9_walls_0.9.txt', 'output_basepath': '../data/results/5_tuple_on_rplan_rooms/temp_0.9_doors_0.9_walls_0.9'},
        # {'box_dir': '../data/results/5_tuple_on_rplan/temp_0.9', 'door_dir': '../data/results/5_tuple_on_rplan/temp_0.9/doors_0.9', 'wall_dir': '../data/results/5_tuple_on_rplan/temp_0.9/walls_1.0', 'sample_list': '../data/results/5_tuple_on_rplan/temp_0.9/test_doors_0.9_walls_1.0.txt', 'output_basepath': '../data/results/5_tuple_on_rplan_rooms/temp_0.9_doors_0.9_walls_1.0'},
        # {'box_dir': '../data/results/5_tuple_on_rplan/temp_0.9', 'door_dir': '../data/results/5_tuple_on_rplan/temp_0.9/doors_1.0', 'wall_dir': '../data/results/5_tuple_on_rplan/temp_0.9/walls_0.9', 'sample_list': '../data/results/5_tuple_on_rplan/temp_0.9/test_doors_1.0_walls_0.9.txt', 'output_basepath': '../data/results/5_tuple_on_rplan_rooms/temp_0.9_doors_1.0_walls_0.9'},
        # {'box_dir': '../data/results/5_tuple_on_rplan/temp_0.9', 'door_dir': '../data/results/5_tuple_on_rplan/temp_0.9/doors_1.0', 'wall_dir': '../data/results/5_tuple_on_rplan/temp_0.9/walls_1.0', 'sample_list': '../data/results/5_tuple_on_rplan/temp_0.9/test_doors_1.0_walls_1.0.txt', 'output_basepath': '../data/results/5_tuple_on_rplan_rooms/temp_0.9_doors_1.0_walls_1.0'},
        # {'box_dir': '../data/results/5_tuple_on_rplan/temp_1.0', 'door_dir': '../data/results/5_tuple_on_rplan/temp_1.0/doors_0.9', 'wall_dir': '../data/results/5_tuple_on_rplan/temp_1.0/walls_0.9', 'sample_list': '../data/results/5_tuple_on_rplan/temp_1.0/test_doors_0.9_walls_0.9.txt', 'output_basepath': '../data/results/5_tuple_on_rplan_rooms/temp_1.0_doors_0.9_walls_0.9'},
        # {'box_dir': '../data/results/5_tuple_on_rplan/temp_1.0', 'door_dir': '../data/results/5_tuple_on_rplan/temp_1.0/doors_0.9', 'wall_dir': '../data/results/5_tuple_on_rplan/temp_1.0/walls_1.0', 'sample_list': '../data/results/5_tuple_on_rplan/temp_1.0/test_doors_0.9_walls_1.0.txt', 'output_basepath': '../data/results/5_tuple_on_rplan_rooms/temp_1.0_doors_0.9_walls_1.0'},
        # {'box_dir': '../data/results/5_tuple_on_rplan/temp_1.0', 'door_dir': '../data/results/5_tuple_on_rplan/temp_1.0/doors_1.0', 'wall_dir': '../data/results/5_tuple_on_rplan/temp_1.0/walls_0.9', 'sample_list': '../data/results/5_tuple_on_rplan/temp_1.0/test_doors_1.0_walls_0.9.txt', 'output_basepath': '../data/results/5_tuple_on_rplan_rooms/temp_1.0_doors_1.0_walls_0.9'},
        # {'box_dir': '../data/results/5_tuple_on_rplan/temp_1.0', 'door_dir': '../data/results/5_tuple_on_rplan/temp_1.0/doors_1.0', 'wall_dir': '../data/results/5_tuple_on_rplan/temp_1.0/walls_1.0', 'sample_list': '../data/results/5_tuple_on_rplan/temp_1.0/test_doors_1.0_walls_1.0.txt', 'output_basepath': '../data/results/5_tuple_on_rplan_rooms/temp_1.0_doors_1.0_walls_1.0'},

        # {'box_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9', 'door_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9/doors_0.9', 'wall_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9/walls_0.9', 'sample_list': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9/test_doors_0.9_walls_0.9.txt', 'output_basepath': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_rooms/temp_0.9_doors_0.9_walls_0.9'},
        # {'box_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9', 'door_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9/doors_0.9', 'wall_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9/walls_1.0', 'sample_list': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9/test_doors_0.9_walls_1.0.txt', 'output_basepath': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_rooms/temp_0.9_doors_0.9_walls_1.0'},
        # {'box_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9', 'door_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9/doors_1.0', 'wall_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9/walls_0.9', 'sample_list': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9/test_doors_1.0_walls_0.9.txt', 'output_basepath': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_rooms/temp_0.9_doors_1.0_walls_0.9'},
        # {'box_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9', 'door_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9/doors_1.0', 'wall_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9/walls_1.0', 'sample_list': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9/test_doors_1.0_walls_1.0.txt', 'output_basepath': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_rooms/temp_0.9_doors_1.0_walls_1.0'},
        # {'box_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0', 'door_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0/doors_0.9', 'wall_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0/walls_0.9', 'sample_list': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0/test_doors_0.9_walls_0.9.txt', 'output_basepath': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_rooms/temp_1.0_doors_0.9_walls_0.9'},
        # {'box_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0', 'door_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0/doors_0.9', 'wall_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0/walls_1.0', 'sample_list': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0/test_doors_0.9_walls_1.0.txt', 'output_basepath': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_rooms/temp_1.0_doors_0.9_walls_1.0'},
        # {'box_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0', 'door_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0/doors_1.0', 'wall_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0/walls_0.9', 'sample_list': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0/test_doors_1.0_walls_0.9.txt', 'output_basepath': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_rooms/temp_1.0_doors_1.0_walls_0.9'},
        # {'box_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0', 'door_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0/doors_1.0', 'wall_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0/walls_1.0', 'sample_list': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0/test_doors_1.0_walls_1.0.txt', 'output_basepath': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_rooms/temp_1.0_doors_1.0_walls_1.0'},
        
        {'box_dir': '../data/results/3_tuple_on_rplan/temp_0.9_0.9/nodes_0.9_0.9.hdf5', 'door_dir': '../data/results/3_tuple_on_rplan/temp_0.9_0.9/doors_0.9.hdf5', 'wall_dir': '../data/results/3_tuple_on_rplan/temp_0.9_0.9/walls_0.9.hdf5', 'sample_list': '../data/results/3_tuple_on_rplan/temp_0.9_0.9/test_doors_0.9_walls_0.9.txt', 'output_basepath': '../data/results/3_tuple_on_rplan_rooms/nodes_0.9_0.9_doors_0.9_walls_0.9'},

        # {'box_dir': '../data/results/rplan_on_rplan', 'sample_list': '../data/results/rplan_on_rplan/all.txt', 'output_basepath': '../data/results/rplan_on_rplan_rooms/rplan_on_rplan'},
        # {'box_dir': '../data/results/rplan_on_lifull', 'sample_list': '../data/results/rplan_on_lifull/all.txt', 'output_basepath': '../data/results/rplan_on_lifull_rooms/rplan_on_lifull'},
        
        # {'box_dir': '/home/guerrero/scratch_space/floorplan/rplan_ddg_var', 'sample_list': '/home/guerrero/scratch_space/floorplan/rplan_ddg_var/test.txt', 'suffix': '_image_nodoor', 'output_basepath': '../data/results/gt_on_rplan_rooms/gt_on_rplan'},
        # {'box_dir': '/home/guerrero/scratch_space/floorplan/lifull_ddg_var', 'sample_list': '/home/guerrero/scratch_space/floorplan/lifull_ddg_var/test.txt', 'suffix': '', 'output_basepath': '../data/results/gt_on_lifull_rooms/gt_on_lifull'},
    ]

    for rsi, result_set in enumerate(result_sets):

        box_dir = result_set['box_dir']
        door_dir = result_set['door_dir'] if 'door_dir' in result_set else None
        wall_dir = result_set['wall_dir'] if 'wall_dir' in result_set else None
        sample_list = result_set['sample_list'] if 'sample_list' in result_set else None
        suffix = result_set['suffix'] if 'suffix' in result_set else ''
        output_basepath = result_set['output_basepath']

        print(f'result set [{rsi+1}/{len(result_sets)}]: {output_basepath}')

        # read the boxes and edges of all floor plans in the input directory
        sample_names = get_box_sample_names(box_dir=box_dir, door_dir=door_dir, wall_dir=wall_dir, sample_list_path=sample_list)

        os.makedirs(os.path.dirname(output_basepath), exist_ok=True)

        batch_count = 0
        batch_size = 100
        batch_count = math.ceil(len(sample_names) / batch_size)
        batch_sample_names, batch_room_types, batch_room_bboxes, batch_room_door_edges, batch_room_door_regions, batch_room_idx_maps = [], [], [], [], [], []
        for batch_idx in tqdm(range(batch_count)):

            samples_from = batch_size*batch_idx
            samples_to = min(batch_size*(batch_idx+1), len(sample_names))
            batch_sample_names = sample_names[samples_from:samples_to]
            
            boxes, door_edges, wall_edges, _ = load_boxes(sample_names=batch_sample_names, box_dir=box_dir, door_dir=door_dir, wall_dir=wall_dir, suffix=suffix)

            room_types, room_bboxes, room_door_edges, room_door_regions, room_idx_maps = convert_boxes_to_rooms(
                boxes=boxes, door_edges=door_edges, wall_edges=wall_edges, img_res=(64, 64), room_type_count=len(room_type_names))

            save_rooms(
                base_path=output_basepath, sample_names=batch_sample_names,
                room_types=room_types,
                room_bboxes=room_bboxes,
                room_door_edges=room_door_edges,
                room_door_regions=room_door_regions,
                room_idx_maps=room_idx_maps,
                append=batch_idx > 0)
