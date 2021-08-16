import os
import math
from load_boxes import get_box_sample_names, load_boxes
from convert_boxes_to_rooms import convert_boxes_to_rooms, room_type_names, BadBoxesException
from tqdm import tqdm

result_sets = [
    # {'box_dir': '../data/results/5_tuple_on_rplan/temp_0.9', 'door_dir': '../data/results/5_tuple_on_rplan/temp_0.9/doors_0.9', 'wall_dir': '../data/results/5_tuple_on_rplan/temp_0.9/walls_0.9', 'output_list': '../data/results/5_tuple_on_rplan/temp_0.9/test_doors_0.9_walls_0.9.txt'},
    # {'box_dir': '../data/results/5_tuple_on_rplan/temp_0.9', 'door_dir': '../data/results/5_tuple_on_rplan/temp_0.9/doors_0.9', 'wall_dir': '../data/results/5_tuple_on_rplan/temp_0.9/walls_1.0', 'output_list': '../data/results/5_tuple_on_rplan/temp_0.9/test_doors_0.9_walls_1.0.txt'},
    # {'box_dir': '../data/results/5_tuple_on_rplan/temp_0.9', 'door_dir': '../data/results/5_tuple_on_rplan/temp_0.9/doors_1.0', 'wall_dir': '../data/results/5_tuple_on_rplan/temp_0.9/walls_0.9', 'output_list': '../data/results/5_tuple_on_rplan/temp_0.9/test_doors_1.0_walls_0.9.txt'},
    # {'box_dir': '../data/results/5_tuple_on_rplan/temp_0.9', 'door_dir': '../data/results/5_tuple_on_rplan/temp_0.9/doors_1.0', 'wall_dir': '../data/results/5_tuple_on_rplan/temp_0.9/walls_1.0', 'output_list': '../data/results/5_tuple_on_rplan/temp_0.9/test_doors_1.0_walls_1.0.txt'},
    # {'box_dir': '../data/results/5_tuple_on_rplan/temp_1.0', 'door_dir': '../data/results/5_tuple_on_rplan/temp_1.0/doors_0.9', 'wall_dir': '../data/results/5_tuple_on_rplan/temp_1.0/walls_0.9', 'output_list': '../data/results/5_tuple_on_rplan/temp_1.0/test_doors_0.9_walls_0.9.txt'},
    # {'box_dir': '../data/results/5_tuple_on_rplan/temp_1.0', 'door_dir': '../data/results/5_tuple_on_rplan/temp_1.0/doors_0.9', 'wall_dir': '../data/results/5_tuple_on_rplan/temp_1.0/walls_1.0', 'output_list': '../data/results/5_tuple_on_rplan/temp_1.0/test_doors_0.9_walls_1.0.txt'},
    # {'box_dir': '../data/results/5_tuple_on_rplan/temp_1.0', 'door_dir': '../data/results/5_tuple_on_rplan/temp_1.0/doors_1.0', 'wall_dir': '../data/results/5_tuple_on_rplan/temp_1.0/walls_0.9', 'output_list': '../data/results/5_tuple_on_rplan/temp_1.0/test_doors_1.0_walls_0.9.txt'},
    # {'box_dir': '../data/results/5_tuple_on_rplan/temp_1.0', 'door_dir': '../data/results/5_tuple_on_rplan/temp_1.0/doors_1.0', 'wall_dir': '../data/results/5_tuple_on_rplan/temp_1.0/walls_1.0', 'output_list': '../data/results/5_tuple_on_rplan/temp_1.0/test_doors_1.0_walls_1.0.txt'},

    # {'box_dir': '../data/results/3_tuple_on_rplan/temp_0.9/nodes_0.9_0.9', 'door_dir': '../data/results/3_tuple_on_rplan/temp_0.9/doors_0.9', 'wall_dir': '../data/results/3_tuple_on_rplan/temp_0.9/walls_0.9', 'output_list': '../data/results/3_tuple_on_rplan/temp_0.9/test_doors_0.9_walls_0.9.txt', 'coord_type': 'normalized'},
    # {'box_dir': '../data/results/3_tuple_on_lifull/temp_0.9/nodes_0.9_0.9', 'door_dir': '../data/results/3_tuple_on_lifull/temp_0.9/doors/_1.0', 'wall_dir': '../data/results/3_tuple_on_lifull/temp_0.9/walls/_1.0', 'output_list': '../data/results/3_tuple_on_lifull/temp_0.9/test_doors_1.0_walls_1.0.txt'},
    # # {'box_dir': '../data/results/3_tuple_on_rplan/temp_0.9_0.9_post_edges/nodes_0.9_0.9.hdf5', 'door_dir': '../data/results/3_tuple_on_rplan/temp_0.9_0.9_post_edges/doors_0.9.hdf5', 'wall_dir': '../data/results/3_tuple_on_rplan/temp_0.9_0.9_post_edges/walls_0.9.hdf5', 'output_list': '../data/results/3_tuple_on_rplan/temp_0.9_0.9_post_edges/test_doors_0.9_walls_0.9.txt'},

    # {'box_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9', 'door_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9/doors_0.9', 'wall_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9/walls_0.9', 'output_list': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9/test_doors_0.9_walls_0.9.txt'},
    # {'box_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9', 'door_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9/doors_0.9', 'wall_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9/walls_1.0', 'output_list': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9/test_doors_0.9_walls_1.0.txt'},
    # {'box_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9', 'door_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9/doors_1.0', 'wall_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9/walls_0.9', 'output_list': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9/test_doors_1.0_walls_0.9.txt'},
    # {'box_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9', 'door_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9/doors_1.0', 'wall_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9/walls_1.0', 'output_list': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9/test_doors_1.0_walls_1.0.txt'},
    # {'box_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0', 'door_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0/doors_0.9', 'wall_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0/walls_0.9', 'output_list': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0/test_doors_0.9_walls_0.9.txt'},
    # {'box_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0', 'door_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0/doors_0.9', 'wall_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0/walls_1.0', 'output_list': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0/test_doors_0.9_walls_1.0.txt'},
    # {'box_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0', 'door_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0/doors_1.0', 'wall_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0/walls_0.9', 'output_list': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0/test_doors_1.0_walls_0.9.txt'},
    # {'box_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0', 'door_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0/doors_1.0', 'wall_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0/walls_1.0', 'output_list': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0/test_doors_1.0_walls_1.0.txt'},

    # {'box_dir': '../data/results/3_tuple_cond_on_rplan/nodes_0.9_merged', 'door_dir': '../data/results/3_tuple_cond_on_rplan/doors_0.9_merged', 'wall_dir': '../data/results/3_tuple_cond_on_rplan/walls_0.9_merged', 'output_list': '../data/results/3_tuple_cond_on_rplan/test_nodes_0.9_doors_0.9_walls_0.9.txt', 'add_exterior': True},
    # {'box_dir': '../data/results/3_tuple_cond_on_lifull/nodes_0.9', 'door_dir': '../data/results/3_tuple_cond_on_lifull/doors_0.9', 'wall_dir': '../data/results/3_tuple_cond_on_lifull/walls_0.9', 'output_list': '../data/results/3_tuple_cond_on_lifull/test_nodes_0.9_doors_0.9_walls_0.9.txt', 'add_exterior': True},

    # {'box_dir': '../data/results/housegan_on_lifull/boxes', 'door_dir': '../data/results/housegan_on_lifull/doors', 'wall_dir': '../data/results/housegan_on_lifull/walls', 'output_list': '../data/results/housegan_on_lifull/all.txt', 'add_exterior': True, 'only_boxes': True, 'coord_type': 'absolute_minmax_corner'},
    {'box_dir': '../data/results/5_tuple_bedroom_cond_on_rplan/nodes/nodes_0.9_0.9', 'door_dir': '../data/results/5_tuple_bedroom_cond_on_rplan/doors', 'wall_dir': '../data/results/5_tuple_bedroom_cond_on_rplan/walls', 'output_list': '../data/results/5_tuple_bedroom_cond_on_rplan/all.txt'},
    {'box_dir': '../data/results/5_tuple_balcony_cond_on_rplan/nodes/nodes_0.9_0.9', 'door_dir': '../data/results/5_tuple_balcony_cond_on_rplan/doors', 'wall_dir': '../data/results/5_tuple_balcony_cond_on_rplan/walls', 'output_list': '../data/results/5_tuple_balcony_cond_on_rplan/all.txt'},
]

max_sample_count = 1000

for rsi, result_set in enumerate(result_sets):

    box_dir = result_set['box_dir']
    door_dir = result_set['door_dir'] if 'door_dir' in result_set else None
    wall_dir = result_set['wall_dir'] if 'wall_dir' in result_set else None
    sample_list = result_set['sample_list'] if 'sample_list' in result_set else None
    suffix = result_set['suffix'] if 'suffix' in result_set else ''
    coord_type = result_set['coord_type'] if 'coord_type' in result_set else 'absolute'
    add_exterior = result_set['add_exterior'] if 'add_exterior' in result_set else False
    only_boxes = result_set['only_boxes'] if 'only_boxes' in result_set else False
    output_list = result_set['output_list']

    print(f'result set [{rsi+1}/{len(result_sets)}]: {output_list}')

    # read the boxes and edges of all floor plans in the input directory
    sample_names = get_box_sample_names(box_dir=box_dir, door_dir=door_dir, wall_dir=wall_dir, sample_list_path=sample_list, only_boxes=only_boxes)

    filter_reasons = {}
    filtered_sample_names = []
    batch_size = 100
    batch_count = math.ceil(len(sample_names) / batch_size)
    pbar = tqdm(range(batch_count))
    for batch_idx in pbar:
        pbar.set_description(f'{len(filtered_sample_names)} filtered samples')

        samples_from = batch_size*batch_idx
        samples_to = min(batch_size*(batch_idx+1), len(sample_names))
        batch_sample_names = sample_names[samples_from:samples_to]

        boxes, door_edges, wall_edges, _ = load_boxes(sample_names=batch_sample_names, box_dir=box_dir, door_dir=door_dir, wall_dir=wall_dir, suffix=suffix, only_boxes=only_boxes)

        for sample_idx, sample_name in enumerate(batch_sample_names):
            try:
                _, _, _, _, _, _, _ = convert_boxes_to_rooms(
                    boxes=[boxes[sample_idx]], door_edges=[door_edges[sample_idx]], wall_edges=[wall_edges[sample_idx]], img_res=(64, 64),
                    room_type_count=len(room_type_names), coord_type=coord_type, add_exterior=add_exterior)
            except BadBoxesException as err:
                # print(f'WARNING: could not parse sample {sample_name}:\n{err}')
                if str(err) not in filter_reasons:
                    filter_reasons[str(err)] = 1
                else:
                    filter_reasons[str(err)] += 1
                continue

            filtered_sample_names.append(sample_name)
            if max_sample_count is not None and len(filtered_sample_names) >= max_sample_count:
                break

        if max_sample_count is not None and len(filtered_sample_names) >= max_sample_count:
            break

    filter_reasons = sorted([(reason, count) for reason, count in filter_reasons.items()], key=lambda x: x[1], reverse=True)
    print('filtered bad samples due to (amount: reason): ')
    for reason, count in filter_reasons:
        print(f'{count}: {reason}')
    
    if len(filtered_sample_names) < max_sample_count:
        print(f'WARNING: could not find enough filtered samples for result set {output_list}')

    os.makedirs(os.path.dirname(output_list), exist_ok=True)
    with open(output_list, 'w') as f:
        for sample_name in filtered_sample_names:
            f.write(f'{sample_name}\n')
