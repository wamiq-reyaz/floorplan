import os
from load_boxes import get_sample_names, load_boxes
from convert_boxes_to_rooms import convert_boxes_to_rooms, room_type_names
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

    # {'box_dir': '../data/results/3_tuple_on_rplan/temp_0.9/nodes_0.9_0.9', 'door_dir': '../data/results/3_tuple_on_rplan/temp_0.9/doors_0.9', 'wall_dir': '../data/results/3_tuple_on_rplan/temp_0.9/walls_0.9', 'output_list': '../data/results/3_tuple_on_rplan/temp_0.9/nodes_0.9_0.9/test_doors_0.9_walls_0.9.txt'},
    # {'box_dir': '../data/results/3_tuple_on_rplan/temp_0.9/nodes_0.9_0.9', 'door_dir': '../data/results/3_tuple_on_rplan/temp_0.9/doors_0.9', 'wall_dir': '../data/results/3_tuple_on_rplan/temp_0.9/walls_1.0', 'output_list': '../data/results/3_tuple_on_rplan/temp_0.9/nodes_0.9_0.9/test_doors_0.9_walls_1.0.txt'},
    # {'box_dir': '../data/results/3_tuple_on_rplan/temp_0.9/nodes_0.9_0.9', 'door_dir': '../data/results/3_tuple_on_rplan/temp_0.9/doors_1.0', 'wall_dir': '../data/results/3_tuple_on_rplan/temp_0.9/walls_0.9', 'output_list': '../data/results/3_tuple_on_rplan/temp_0.9/nodes_0.9_0.9/test_doors_1.0_walls_0.9.txt'},
    # {'box_dir': '../data/results/3_tuple_on_rplan/temp_0.9/nodes_0.9_0.9', 'door_dir': '../data/results/3_tuple_on_rplan/temp_0.9/doors_1.0', 'wall_dir': '../data/results/3_tuple_on_rplan/temp_0.9/walls_1.0', 'output_list': '../data/results/3_tuple_on_rplan/temp_0.9/nodes_0.9_0.9/test_doors_1.0_walls_1.0.txt'},
    # {'box_dir': '../data/results/3_tuple_on_rplan/temp_0.9/nodes_0.9_1.0', 'door_dir': '../data/results/3_tuple_on_rplan/temp_0.9/doors_0.9', 'wall_dir': '../data/results/3_tuple_on_rplan/temp_0.9/walls_0.9', 'output_list': '../data/results/3_tuple_on_rplan/temp_0.9/nodes_0.9_1.0/test_doors_0.9_walls_0.9.txt'},
    # {'box_dir': '../data/results/3_tuple_on_rplan/temp_0.9/nodes_0.9_1.0', 'door_dir': '../data/results/3_tuple_on_rplan/temp_0.9/doors_0.9', 'wall_dir': '../data/results/3_tuple_on_rplan/temp_0.9/walls_1.0', 'output_list': '../data/results/3_tuple_on_rplan/temp_0.9/nodes_0.9_1.0/test_doors_0.9_walls_1.0.txt'},
    # {'box_dir': '../data/results/3_tuple_on_rplan/temp_0.9/nodes_0.9_1.0', 'door_dir': '../data/results/3_tuple_on_rplan/temp_0.9/doors_1.0', 'wall_dir': '../data/results/3_tuple_on_rplan/temp_0.9/walls_0.9', 'output_list': '../data/results/3_tuple_on_rplan/temp_0.9/nodes_0.9_1.0/test_doors_1.0_walls_0.9.txt'},
    # {'box_dir': '../data/results/3_tuple_on_rplan/temp_0.9/nodes_0.9_1.0', 'door_dir': '../data/results/3_tuple_on_rplan/temp_0.9/doors_1.0', 'wall_dir': '../data/results/3_tuple_on_rplan/temp_0.9/walls_1.0', 'output_list': '../data/results/3_tuple_on_rplan/temp_0.9/nodes_0.9_1.0/test_doors_1.0_walls_1.0.txt'},
    # {'box_dir': '../data/results/3_tuple_on_rplan/temp_1.0/nodes_1.0_1.0', 'door_dir': '../data/results/3_tuple_on_rplan/temp_1.0/doors_0.9', 'wall_dir': '../data/results/3_tuple_on_rplan/temp_1.0/walls_0.9', 'output_list': '../data/results/3_tuple_on_rplan/temp_1.0/nodes_1.0_1.0/test_doors_0.9_walls_0.9.txt'},
    # {'box_dir': '../data/results/3_tuple_on_rplan/temp_1.0/nodes_1.0_1.0', 'door_dir': '../data/results/3_tuple_on_rplan/temp_1.0/doors_0.9', 'wall_dir': '../data/results/3_tuple_on_rplan/temp_1.0/walls_1.0', 'output_list': '../data/results/3_tuple_on_rplan/temp_1.0/nodes_1.0_1.0/test_doors_0.9_walls_1.0.txt'},
    # {'box_dir': '../data/results/3_tuple_on_rplan/temp_1.0/nodes_1.0_1.0', 'door_dir': '../data/results/3_tuple_on_rplan/temp_1.0/doors_1.0', 'wall_dir': '../data/results/3_tuple_on_rplan/temp_1.0/walls_0.9', 'output_list': '../data/results/3_tuple_on_rplan/temp_1.0/nodes_1.0_1.0/test_doors_1.0_walls_0.9.txt'},
    # {'box_dir': '../data/results/3_tuple_on_rplan/temp_1.0/nodes_1.0_1.0', 'door_dir': '../data/results/3_tuple_on_rplan/temp_1.0/doors_1.0', 'wall_dir': '../data/results/3_tuple_on_rplan/temp_1.0/walls_1.0', 'output_list': '../data/results/3_tuple_on_rplan/temp_1.0/nodes_1.0_1.0/test_doors_1.0_walls_1.0.txt'},

    {'box_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9', 'door_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9/doors_0.9', 'wall_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9/walls_0.9', 'output_list': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9/test_doors_0.9_walls_0.9.txt'},
    {'box_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9', 'door_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9/doors_0.9', 'wall_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9/walls_1.0', 'output_list': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9/test_doors_0.9_walls_1.0.txt'},
    {'box_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9', 'door_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9/doors_1.0', 'wall_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9/walls_0.9', 'output_list': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9/test_doors_1.0_walls_0.9.txt'},
    {'box_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9', 'door_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9/doors_1.0', 'wall_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9/walls_1.0', 'output_list': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_0.9/test_doors_1.0_walls_1.0.txt'},
    {'box_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0', 'door_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0/doors_0.9', 'wall_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0/walls_0.9', 'output_list': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0/test_doors_0.9_walls_0.9.txt'},
    {'box_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0', 'door_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0/doors_0.9', 'wall_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0/walls_1.0', 'output_list': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0/test_doors_0.9_walls_1.0.txt'},
    {'box_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0', 'door_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0/doors_1.0', 'wall_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0/walls_0.9', 'output_list': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0/test_doors_1.0_walls_0.9.txt'},
    {'box_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0', 'door_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0/doors_1.0', 'wall_dir': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0/walls_1.0', 'output_list': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull/temp_1.0/test_doors_1.0_walls_1.0.txt'},
]

max_sample_count = 1000

for rsi, result_set in enumerate(result_sets):

    box_dir = result_set['box_dir']
    door_dir = result_set['door_dir'] if 'door_dir' in result_set else None
    wall_dir = result_set['wall_dir'] if 'wall_dir' in result_set else None
    sample_list = result_set['sample_list'] if 'sample_list' in result_set else None
    suffix = result_set['suffix'] if 'suffix' in result_set else ''
    output_list = result_set['output_list']

    print(f'result set [{rsi+1}/{len(result_sets)}]: {output_list}')

    # read the boxes and edges of all floor plans in the input directory
    sample_names = get_sample_names(box_dir=box_dir, door_dir=door_dir, wall_dir=wall_dir, sample_list_path=sample_list)

    filtered_sample_names = []
    for sample_name in tqdm(sample_names):
        boxes, door_edges, wall_edges, sample_names = load_boxes(sample_names=[sample_name], box_dir=box_dir, door_dir=door_dir, wall_dir=wall_dir, suffix=suffix)

        try:
            _, _, _, _ = convert_boxes_to_rooms(
                boxes=boxes, door_edges=door_edges, wall_edges=wall_edges, img_res=(64, 64), room_type_count=len(room_type_names))
        except ValueError as err:
            print(f'WARNING: could not parse sample {sample_name}:\n{err}')
            continue

        filtered_sample_names.append(sample_names[0])
        if max_sample_count is not None and len(filtered_sample_names) >= max_sample_count:
            break

    if len(filtered_sample_names) < max_sample_count:
        print(f'WARNING: could not find enough filtered samples for result set {output_list}')

    os.makedirs(os.path.dirname(output_list), exist_ok=True)
    with open(output_list, 'w') as f:
        for sample_name in filtered_sample_names:
            f.write(f'{sample_name}\n')
