import numpy as np
import torch
from skimage import draw

def pixel_coord_meshgrid(size, device):
    pixel_coords_y, pixel_coords_x = torch.meshgrid(
        torch.linspace(0, size[0]-1, steps=size[0], device=device),
        torch.linspace(0, size[1]-1, steps=size[1], device=device))
    pixel_coords = torch.cat([pixel_coords_y.unsqueeze(0), pixel_coords_x.unsqueeze(0)], dim=0)

    return pixel_coords

# masks: (batch_size, height, width)
# out: y, x coords (batch_size, 2)
def mask_centroids(masks, pixel_coords=None):
    if pixel_coords is None:
        pixel_coords = pixel_coord_meshgrid(
            size=[masks.shape[1], masks.shape[2]], device=masks.device)

    centroids = (
        pixel_coords.view(1, 2, pixel_coords.shape[1]*pixel_coords.shape[2]) *
        masks.view(masks.shape[0], 1, -1)
        ).sum(dim=-1) / torch.clamp(masks.view(masks.shape[0], 1, -1).sum(dim=-1), min=1)

    return centroids, pixel_coords

def render_floorplans(room_masks, room_type_idx, room_door_adj, room_type_colors):
    # room_door_adj: an array of tensors of shape: edge_type_count x room_count x room_count
    # room_door_adj: an array of tensors of shape: edge_type_count x room_count x room_count

    if len(room_masks) == 0:
        return []

    # get room and edge colors
    edge_type_colors = torch.tensor([
        [1, 0, 0, 1],
        [0, 0, 1, 0.5],
        [0, 1, 0, 0.5],
    ], dtype=torch.float32, device=room_masks[0].device)
    room_type_colors = room_type_colors.to(room_masks[0].device)

    images = []
    for b in range(len(room_masks)):

        image = (
            room_type_colors[room_type_idx[b], :].view(-1, 3, 1, 1) *
            room_masks[b].view(-1, 1, room_masks[b].shape[1], room_masks[b].shape[2])
            ).sum(dim=0).clamp(min=0, max=1)

        if room_door_adj is not None:

            pixel_coords = pixel_coord_meshgrid(
                size=[room_masks[b].shape[1], room_masks[b].shape[2]], device=room_masks[b].device)

            node_radius = 2

            room_door_adj_triu = room_door_adj[b].triu(diagonal=1)

            node_locations = mask_centroids(room_masks[b], pixel_coords=pixel_coords)[0]
            node_locations = node_locations.round().to(dtype=torch.int64)

            for edge_type_idx in range(room_door_adj_triu.shape[0]):
                edge_color = edge_type_colors[edge_type_idx, :3]
                edge_opacity = edge_type_colors[edge_type_idx, 3]
                edge_list = torch.nonzero(room_door_adj_triu[edge_type_idx], as_tuple=False)

                for edge in edge_list:
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

                    alpha = (torch.from_numpy(line_alpha).to(device=room_masks[b].device, dtype=torch.float32) * edge_opacity).view(1, -1)
                    image[:, rr, cc] = (1-alpha) * image[:, rr, cc] + alpha * edge_color.view(3, 1)

        images.append(image)

    return images

if __name__ == '__main__':
    import os
    from load_boxes import get_sample_names, load_boxes
    from convert_boxes_to_rooms import convert_boxes_to_rooms, room_type_colors
    from node import Floor
    from tqdm import tqdm
    import torchvision
    import math

    # input_dir = '../data/results/5_tuples_t_0.8_minimal'
    # output_dir = '../data/results/5_tuples_t_0.8_minimal/vis'
    # input_list = None
    # suffix = ''

    input_dir = '../data/results/5_tuples_t_0.8'
    output_dir = '../data/results/5_tuples_t_0.8/vis'
    input_list = None
    suffix = ''

    # input_dir = '/home/guerrero/scratch_space/floorplan/rplan_ddg_var'
    # output_dir = '/home/guerrero/scratch_space/floorplan/rplan_ddg_var_vis'
    # input_list = '/home/guerrero/scratch_space/floorplan/rplan_ddg_var/test.txt'
    # suffix='_image_nodoor'

    # input_dir = '../data/results/rplan_var_images_doors'
    # output_dir = '../data/results/rplan_var_images_doors_vis'
    # input_list = '../data/results/rplan_var_images_doors/all.txt'
    # suffix=''

    os.makedirs(output_dir, exist_ok=True)

    # read the boxes and edges of all floor plans in the input directory
    sample_names, sample_format = get_sample_names(input_dir=input_dir, input_list=input_list)

    for sample_name in tqdm(sample_names):
        boxes, door_edges, wall_edges, sample_names = load_boxes(input_dir=input_dir, sample_names=[sample_name], format=sample_format, suffix=suffix)

        try:
            room_types, room_bboxes, room_door_adj, room_masks = convert_boxes_to_rooms(
                boxes=boxes, door_edges=door_edges, wall_edges=wall_edges, img_res=(64, 64), room_type_count=room_type_colors.shape[0])
        except ValueError as err:
            print(f'WARNING: could not parse sample {sample_name}:\n{err}')
            continue

        image = render_floorplans(
            room_masks=room_masks,
            room_type_idx=room_types,
            room_door_adj=[a.unsqueeze(0) for a in room_door_adj],
            room_type_colors=torch.tensor(room_type_colors))

        out_filename = os.path.join(output_dir, f'{sample_name}.png')
        os.makedirs(os.path.dirname(out_filename), exist_ok=True)
        torchvision.utils.save_image(image, out_filename)

    # image = render_floorplans(
    #     room_masks=[room_masks[0]],
    #     room_type_idx=[room_types[0]],
    #     room_door_adj=[room_door_adj[0].unsqueeze(0)],
    #     room_type_colors=torch.tensor(room_type_colors))[0]

    # import torchvision
    # torchvision.utils.save_image(image, '/home/guerrero/scratch_space/floorplan/test_floorplan_render.png')
