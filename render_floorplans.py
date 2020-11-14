import numpy as np
from skimage import draw
from skimage.io import imsave

def one_hot(inp, label_count):
    out = np.zeros([inp.size, label_count], dtype=np.bool)
    out[np.arange(out.shape[0]), inp.reshape(-1)] = True
    out = out.reshape(inp.shape+(label_count,))
    return out

def pixel_coord_meshgrid(size):
    pixel_coords_y, pixel_coords_x = np.meshgrid(
        np.linspace(0, size[0]-1, num=size[0]),
        np.linspace(0, size[1]-1, num=size[1]),
        indexing='ij')
    pixel_coords = np.stack([pixel_coords_y, pixel_coords_x])

    return pixel_coords

# masks: (batch_size, height, width)
# out: y, x coords (batch_size, 2)
def mask_centroids(masks, pixel_coords=None):
    if pixel_coords is None:
        pixel_coords = pixel_coord_meshgrid(
            size=[masks.shape[1], masks.shape[2]])

    centroids = (
        pixel_coords.reshape(1, 2, pixel_coords.shape[1]*pixel_coords.shape[2]) *
        masks.reshape(masks.shape[0], 1, -1)
        ).sum(axis=-1) / masks.reshape(masks.shape[0], 1, -1).sum(axis=-1).clip(min=1, max=None)

    return centroids, pixel_coords

def render_floorplans(room_idx_maps, room_type_idx, room_door_edges, room_type_colors):

    if len(room_idx_maps) == 0:
        return []

    # get room and edge colors
    edge_color = np.array([1, 0, 0])
    edge_opacity = 1.0
    room_type_colors = room_type_colors

    images = []
    for b in range(len(room_idx_maps)):
        room_masks = one_hot(room_idx_maps[b], label_count=room_type_idx[b].size).transpose(2, 0, 1)

        image = (
            room_type_colors[room_type_idx[b], :].reshape(-1, 3, 1, 1) *
            room_masks.reshape(-1, 1, room_masks.shape[1], room_masks.shape[2])
            ).sum(axis=0).clip(min=0, max=1)

        if room_door_edges is not None:
            pixel_coords = pixel_coord_meshgrid(size=[room_masks.shape[1], room_masks.shape[2]])

            node_radius = 2

            node_locations = mask_centroids(room_masks, pixel_coords=pixel_coords)[0]
            node_locations = node_locations.round().astype(np.int64)

            for edge in room_door_edges[b]:
                edge_start = node_locations[edge[0], :]
                edge_end = node_locations[edge[1], :]

                if room_type_idx[b][edge[0]] == 0 and room_type_idx[b][edge[1]] == 0:
                    # edge between two exterior nodes (don't draw anything)
                    continue
                elif room_type_idx[b][edge[0]] == 0 or room_type_idx[b][edge[1]] == 0:
                    # edge between exterior and non-exterior node (draw circle around non-exterior node)
                    if room_type_idx[b][edge[0]] != 0:
                        [rr, cc, line_alpha] = draw.circle_perimeter_aa(edge_start[0], edge_start[1], node_radius+1)
                    elif room_type_idx[b][edge[1]] != 0:
                        [rr, cc, line_alpha] = draw.circle_perimeter_aa(edge_end[0], edge_end[1], node_radius+1)
                else:
                    # edge between non-exterior nodes (draw line between them)
                    [rr, cc, line_alpha] = draw.line_aa(edge_start[0], edge_start[1], edge_end[0], edge_end[1])

                # cull values outside the image
                mask = np.logical_and(np.logical_and(np.logical_and(rr >= 0, rr < image.shape[1]), cc >= 0), cc < image.shape[2])
                rr = rr[mask]
                cc = cc[mask]
                line_alpha = line_alpha[mask]

                alpha = (line_alpha.astype(np.float32) * edge_opacity).reshape(1, -1)
                image[:, rr, cc] = (1-alpha) * image[:, rr, cc] + alpha * edge_color.reshape(3, 1)

        images.append(image.clip(min=0.0, max=1.0))

    return images

if __name__ == '__main__':
    import os
    import math
    # from load_boxes import get_box_sample_names, load_boxes
    from load_boxes import get_room_sample_names, load_rooms
    from convert_boxes_to_rooms import convert_boxes_to_rooms, room_type_colors
    from tqdm import tqdm
    import torchvision

    result_sets = [
        # {'room_basepath': '../data/results/5_tuple_on_rplan_rooms/temp_0.9_doors_0.9_walls_0.9', 'output_dir': '../data/results/5_tuple_on_rplan_vis/temp_0.9_doors_0.9_walls_0.9'},
        # {'room_basepath': '../data/results/5_tuple_on_rplan_rooms/temp_0.9_doors_0.9_walls_1.0', 'output_dir': '../data/results/5_tuple_on_rplan_vis/temp_0.9_doors_0.9_walls_1.0'},
        # {'room_basepath': '../data/results/5_tuple_on_rplan_rooms/temp_0.9_doors_1.0_walls_0.9', 'output_dir': '../data/results/5_tuple_on_rplan_vis/temp_0.9_doors_1.0_walls_0.9'},
        # {'room_basepath': '../data/results/5_tuple_on_rplan_rooms/temp_0.9_doors_1.0_walls_1.0', 'output_dir': '../data/results/5_tuple_on_rplan_vis/temp_0.9_doors_1.0_walls_1.0'},
        # {'room_basepath': '../data/results/5_tuple_on_rplan_rooms/temp_1.0_doors_0.9_walls_0.9', 'output_dir': '../data/results/5_tuple_on_rplan_vis/temp_1.0_doors_0.9_walls_0.9'},
        # {'room_basepath': '../data/results/5_tuple_on_rplan_rooms/temp_1.0_doors_0.9_walls_1.0', 'output_dir': '../data/results/5_tuple_on_rplan_vis/temp_1.0_doors_0.9_walls_1.0'},
        # {'room_basepath': '../data/results/5_tuple_on_rplan_rooms/temp_1.0_doors_1.0_walls_0.9', 'output_dir': '../data/results/5_tuple_on_rplan_vis/temp_1.0_doors_1.0_walls_0.9'},
        # {'room_basepath': '../data/results/5_tuple_on_rplan_rooms/temp_1.0_doors_1.0_walls_1.0', 'output_dir': '../data/results/5_tuple_on_rplan_vis/temp_1.0_doors_1.0_walls_1.0'},

        # {'room_basepath': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_rooms/temp_0.9_doors_0.9_walls_0.9', 'output_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_vis/temp_0.9_doors_0.9_walls_0.9'},
        # {'room_basepath': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_rooms/temp_0.9_doors_0.9_walls_1.0', 'output_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_vis/temp_0.9_doors_0.9_walls_1.0'},
        # {'room_basepath': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_rooms/temp_0.9_doors_1.0_walls_0.9', 'output_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_vis/temp_0.9_doors_1.0_walls_0.9'},
        # {'room_basepath': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_rooms/temp_0.9_doors_1.0_walls_1.0', 'output_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_vis/temp_0.9_doors_1.0_walls_1.0'},
        # {'room_basepath': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_rooms/temp_1.0_doors_0.9_walls_0.9', 'output_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_vis/temp_1.0_doors_0.9_walls_0.9'},
        # {'room_basepath': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_rooms/temp_1.0_doors_0.9_walls_1.0', 'output_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_vis/temp_1.0_doors_0.9_walls_1.0'},
        # {'room_basepath': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_rooms/temp_1.0_doors_1.0_walls_0.9', 'output_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_vis/temp_1.0_doors_1.0_walls_0.9'},
        # {'room_basepath': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_rooms/temp_1.0_doors_1.0_walls_1.0', 'output_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_vis/temp_1.0_doors_1.0_walls_1.0'},

        # {'room_basepath': '../data/results/3_tuple_on_rplan_rooms/nodes_0.9_0.9_doors_0.9_walls_0.9', 'output_dir': '../data/results/3_tuple_on_rplan_vis/nodes_0.9_0.9_doors_0.9_walls_0.9'},
        # {'room_basepath': '../data/results/3_tuple_on_rplan_rooms/nodes_0.9_0.9_doors_0.9_walls_0.9_post_edges', 'output_dir': '../data/results/3_tuple_on_rplan_vis/nodes_0.9_0.9_doors_0.9_walls_0.9_post_edges'},

        # {'room_basepath': '../data/results/rplan_on_rplan_rooms/rplan_on_rplan', 'output_dir': '../data/results/rplan_on_rplan_vis'},
        {'room_basepath': '../data/results/rplan_on_lifull_rooms/rplan_on_lifull', 'output_dir': '../data/results/rplan_on_lifull_vis'},
        
        # {'room_basepath': '../data/results/stylegan_on_rplan_rooms/stylegan_on_rplan', 'output_dir': '../data/results/stylegan_on_rplan_vis'},
        # {'room_basepath': '../data/results/stylegan_on_lifull_rooms/stylegan_on_lifull', 'output_dir': '../data/results/stylegan_on_lifull_vis'},

        # {'room_basepath': '../data/results/graph2plan_on_rplan_rooms/graph2plan_on_rplan', 'output_dir': '../data/results/graph2plan_on_rplan_vis'},
        # {'room_basepath': '../data/results/graph2plan_on_lifull_rooms/graph2plan_on_lifull', 'output_dir': '../data/results/graph2plan_on_lifull_vis'},

        # {'room_basepath': '../data/results/gt_on_rplan_rooms/gt_on_rplan', 'output_dir': '../data/results/gt_on_rplan_vis'},
        # {'room_basepath': '../data/results/gt_on_lifull_rooms/gt_on_lifull', 'output_dir': '../data/results/gt_on_lifull_vis'},
    ]
    
    for rsi, result_set in enumerate(result_sets):

        room_basepath = result_set['room_basepath']
        output_dir = result_set['output_dir']

        print(f'result set [{rsi+1}/{len(result_sets)}]: {output_dir}')
    
        # read the boxes and edges of all floor plans in the input directory
        sample_names = get_room_sample_names(base_path=room_basepath)

        os.makedirs(output_dir, exist_ok=True)

        batch_size = 100
        batch_count = math.ceil(len(sample_names) / batch_size)
        for batch_idx in tqdm(range(batch_count)):
            samples_from = batch_size*batch_idx
            samples_to = min(batch_size*(batch_idx+1), len(sample_names))
            batch_sample_names = sample_names[samples_from:samples_to]

            room_types, _, room_door_edges, _, room_idx_maps, _, _ = load_rooms(
                base_path=room_basepath, sample_names=batch_sample_names)

            images = render_floorplans(
                room_idx_maps=room_idx_maps,
                room_type_idx=room_types,
                room_door_edges=room_door_edges,
                room_type_colors=room_type_colors)

            for sample_idx, sample_name in enumerate(batch_sample_names):
                out_filename = os.path.join(output_dir, f'{sample_name}.png')
                os.makedirs(os.path.dirname(out_filename), exist_ok=True)
                imsave(out_filename, (images[sample_idx].transpose(1, 2, 0)*255.0).astype(np.uint8))
