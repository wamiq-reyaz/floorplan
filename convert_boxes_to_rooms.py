from itertools import chain
import numpy as np
import torch
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
    'Garage',
    'Balcony',
    'Hallway',
    'Other Room']

def convert_boxes_to_rooms(boxes, door_edges, wall_edges, img_res, room_type_count):

    # convert all the boxes in all floor plans to rooms
    room_types, room_bboxes, room_door_adj, room_masks = [], [], [], []
    for fp_boxes, fp_door_edges, fp_wall_edges in zip(boxes, door_edges, wall_edges):

        if fp_boxes.ndim != 2 or fp_boxes.shape[1] != 5:
            raise ValueError('Invalid boxes, boxes array has incorrect shape.')

        if fp_door_edges.ndim != 2 or fp_door_edges.shape[1] != 2 or fp_wall_edges.ndim != 2 or fp_wall_edges.shape[1] != 2:
            raise ValueError('Invalid edges, edges array has incorrect shape.')

        if fp_boxes.size == 0:
            raise ValueError('Empty boxes.')

        if (fp_door_edges.size > 0  and fp_door_edges.max() >= fp_boxes.shape[0]) or (fp_wall_edges.size > 0 and fp_wall_edges.max() >= fp_boxes.shape[0]):
            raise ValueError('Invalid edges, the index is out of bounds.')

        if fp_boxes.dtype == np.float32:
            # boxes are probably 3-tuples which are stored in float format, with the x,y,w,h coordinates from [0...1] instead of [0..img_res-1]
            fp_boxes[:, [1, 3]] *= img_res[1]-1
            fp_boxes[:, [2, 4]] *= img_res[0]-1
            fp_boxes = fp_boxes.round().astype(np.int64)

        if fp_boxes.size > 0 and (fp_boxes[:, 0].max() >= room_type_count or fp_boxes[:, 1].max() >= img_res[1] or fp_boxes[:, 2].max() >= img_res[0] or (fp_boxes[:, 1]+fp_boxes[:, 3]).max() > img_res[1] or (fp_boxes[:, 2]+fp_boxes[:, 4]).max() > img_res[0]):
            raise ValueError('Invalid boxes, the type index or box shapes are out of bounds.')

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

        # get connected copmonents of graph with same room edges; these form the rooms
        same_room_graph = nx.Graph()
        same_room_graph.add_nodes_from(list(range(fp_boxes.shape[0])))
        same_room_graph.add_edges_from(same_room_edges.tolist())
        same_room_conn_comp = [list(comp) for comp in nx.connected_components(same_room_graph)]

        # merge all components of type 'exterior' (with type index 1) into a single component
        exterior_comp = [comp for comp in same_room_conn_comp if fp_boxes[comp[0], 0] == 0]
        nonexterior_comp = [comp for comp in same_room_conn_comp if fp_boxes[comp[0], 0] != 0]
        same_room_conn_comp = [list(chain(*exterior_comp))] + nonexterior_comp

        # create room masks, types, bounding boxes, and door edges
        fp_room_types = np.zeros([len(same_room_conn_comp)], dtype=np.int64)
        fp_room_bboxes = np.zeros([len(same_room_conn_comp), 4], dtype=np.int64)
        fp_room_door_adj = np.zeros([len(same_room_conn_comp), len(same_room_conn_comp)], dtype=np.bool)
        fp_room_masks = np.zeros([len(same_room_conn_comp), *img_res], dtype=np.bool)
        for comp_ind, comp in enumerate(same_room_conn_comp):

            if np.unique(fp_boxes[comp, 0]).size != 1:
                raise ValueError('Found set of boxes with mixed types that are connected by same room edges.')
                # this should not happen since I never generate same room edges between boxes of different type

            # set room type
            fp_room_types[comp_ind] = fp_boxes[comp[0], 0]

            fp_room_bboxes[comp_ind, :] = fp_boxes[comp[0], 1:]
            for box_ind in comp:
                # update room mask
                fp_room_masks[
                    comp_ind,
                    fp_boxes[box_ind, 2]:fp_boxes[box_ind, 2]+fp_boxes[box_ind, 4],
                    fp_boxes[box_ind, 1]:fp_boxes[box_ind, 1]+fp_boxes[box_ind, 3]] = True

                # update room bounding box
                fp_room_bboxes[comp_ind, :2] = np.minimum(fp_room_bboxes[comp_ind, :2], fp_boxes[box_ind, 1:3])
                fp_room_bboxes[comp_ind, 2:] = np.maximum(fp_boxes[box_ind, 1:3]+fp_boxes[box_ind, 3:5]-fp_room_bboxes[comp_ind, :2], fp_room_bboxes[comp_ind, 2:] - fp_room_bboxes[comp_ind, :2])

            # update room door edges (room door edge probability is the maximum probability of any edge that connects any box in room 1 to any box in room 2)
            for comp_ind2, comp2 in enumerate(same_room_conn_comp):
                fp_room_door_adj[comp_ind, comp_ind2] = door_adj[comp, :][:, comp2].any()
        
        room_types.append(torch.tensor(np.transpose(fp_room_types)))
        room_bboxes.append(torch.tensor(np.transpose(fp_room_bboxes)))
        room_door_adj.append(torch.tensor(fp_room_door_adj))
        room_masks.append(torch.tensor(fp_room_masks))

    return room_types, room_bboxes, room_door_adj, room_masks
