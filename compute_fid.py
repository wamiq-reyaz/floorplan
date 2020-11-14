import os
import math
import numpy as np
from tqdm import tqdm

from fid.fid_score import get_model, compute_activations_for_images, compute_statistics, calculate_frechet_distance
from render_floorplans import render_floorplans
from load_boxes import get_room_sample_names, load_rooms
from convert_boxes_to_rooms import room_type_colors

if __name__ == '__main__':

    compute_stats = True
    compute_stat_distances = True

    if compute_stats:

        result_sets = [
            # {'room_basepath': '../data/results/5_tuple_on_rplan_rooms/temp_0.9_doors_0.9_walls_0.9', 'out_filename': '../data/results/5_tuple_on_rplan_stats/temp_0.9_doors_0.9_walls_0.9/inceptionv3_stats.npz'},
            # {'room_basepath': '../data/results/5_tuple_on_rplan_rooms/temp_0.9_doors_0.9_walls_1.0', 'out_filename': '../data/results/5_tuple_on_rplan_stats/temp_0.9_doors_0.9_walls_1.0/inceptionv3_stats.npz'},
            # {'room_basepath': '../data/results/5_tuple_on_rplan_rooms/temp_0.9_doors_1.0_walls_0.9', 'out_filename': '../data/results/5_tuple_on_rplan_stats/temp_0.9_doors_1.0_walls_0.9/inceptionv3_stats.npz'},
            # {'room_basepath': '../data/results/5_tuple_on_rplan_rooms/temp_0.9_doors_1.0_walls_1.0', 'out_filename': '../data/results/5_tuple_on_rplan_stats/temp_0.9_doors_1.0_walls_1.0/inceptionv3_stats.npz'},
            # {'room_basepath': '../data/results/5_tuple_on_rplan_rooms/temp_1.0_doors_0.9_walls_0.9', 'out_filename': '../data/results/5_tuple_on_rplan_stats/temp_1.0_doors_0.9_walls_0.9/inceptionv3_stats.npz'},
            # {'room_basepath': '../data/results/5_tuple_on_rplan_rooms/temp_1.0_doors_0.9_walls_1.0', 'out_filename': '../data/results/5_tuple_on_rplan_stats/temp_1.0_doors_0.9_walls_1.0/inceptionv3_stats.npz'},
            # {'room_basepath': '../data/results/5_tuple_on_rplan_rooms/temp_1.0_doors_1.0_walls_0.9', 'out_filename': '../data/results/5_tuple_on_rplan_stats/temp_1.0_doors_1.0_walls_0.9/inceptionv3_stats.npz'},
            # {'room_basepath': '../data/results/5_tuple_on_rplan_rooms/temp_1.0_doors_1.0_walls_1.0', 'out_filename': '../data/results/5_tuple_on_rplan_stats/temp_1.0_doors_1.0_walls_1.0/inceptionv3_stats.npz'},

            # {'room_basepath': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_rooms/temp_0.9_doors_0.9_walls_0.9', 'out_filename': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_stats/temp_0.9_doors_0.9_walls_0.9/inceptionv3_stats.npz'},
            # {'room_basepath': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_rooms/temp_0.9_doors_0.9_walls_1.0', 'out_filename': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_stats/temp_0.9_doors_0.9_walls_1.0/inceptionv3_stats.npz'},
            # {'room_basepath': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_rooms/temp_0.9_doors_1.0_walls_0.9', 'out_filename': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_stats/temp_0.9_doors_1.0_walls_0.9/inceptionv3_stats.npz'},
            # {'room_basepath': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_rooms/temp_0.9_doors_1.0_walls_1.0', 'out_filename': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_stats/temp_0.9_doors_1.0_walls_1.0/inceptionv3_stats.npz'},
            # {'room_basepath': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_rooms/temp_1.0_doors_0.9_walls_0.9', 'out_filename': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_stats/temp_1.0_doors_0.9_walls_0.9/inceptionv3_stats.npz'},
            # {'room_basepath': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_rooms/temp_1.0_doors_0.9_walls_1.0', 'out_filename': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_stats/temp_1.0_doors_0.9_walls_1.0/inceptionv3_stats.npz'},
            # {'room_basepath': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_rooms/temp_1.0_doors_1.0_walls_0.9', 'out_filename': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_stats/temp_1.0_doors_1.0_walls_0.9/inceptionv3_stats.npz'},
            # {'room_basepath': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_rooms/temp_1.0_doors_1.0_walls_1.0', 'out_filename': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_stats/temp_1.0_doors_1.0_walls_1.0/inceptionv3_stats.npz'},

            # {'room_basepath': '../data/results/3_tuple_on_rplan_rooms/nodes_0.9_0.9_doors_0.9_walls_0.9', 'out_filename': '../data/results/3_tuple_on_rplan_stats/nodes_0.9_0.9_doors_0.9_walls_0.9/inceptionv3_stats.npz'},
            # {'room_basepath': '../data/results/3_tuple_on_lifull_rooms/nodes_0.9_0.9_doors_1.0_walls_1.0', 'out_filename': '../data/results/3_tuple_on_lifull_stats/nodes_0.9_0.9_doors_1.0_walls_1.0/inceptionv3_stats.npz'},
            # # {'room_basepath': '../data/results/3_tuple_on_rplan_rooms/nodes_0.9_0.9_doors_0.9_walls_0.9_post_edges', 'out_filename': '../data/results/3_tuple_on_rplan_stats/nodes_0.9_0.9_doors_0.9_walls_0.9_post_edges/inceptionv3_stats.npz'},

            # {'room_basepath': '../data/results/rplan_on_rplan_rooms/rplan_on_rplan', 'out_filename': '../data/results/rplan_on_rplan_stats/inceptionv3_stats.npz'},
            {'room_basepath': '../data/results/rplan_on_lifull_rooms/rplan_on_lifull', 'out_filename': '../data/results/rplan_on_lifull_stats/inceptionv3_stats.npz'},

            # {'room_basepath': '../data/results/stylegan_on_rplan_rooms/stylegan_on_rplan', 'out_filename': '../data/results/stylegan_on_rplan_stats/inceptionv3_stats.npz'},
            # {'room_basepath': '../data/results/stylegan_on_lifull_rooms/stylegan_on_lifull', 'out_filename': '../data/results/stylegan_on_lifull_stats/inceptionv3_stats.npz'},

            # {'room_basepath': '../data/results/graph2plan_on_rplan_rooms/graph2plan_on_rplan', 'out_filename': '../data/results/graph2plan_on_rplan_stats/inceptionv3_stats.npz'},
            # {'room_basepath': '../data/results/graph2plan_on_lifull_rooms/graph2plan_on_lifull', 'out_filename': '../data/results/graph2plan_on_lifull_stats/inceptionv3_stats.npz'},

            # {'room_basepath': '../data/results/gt_on_rplan_rooms/gt_on_rplan', 'out_filename': '../data/results/gt_on_rplan_stats/inceptionv3_stats.npz'},
            # {'room_basepath': '../data/results/gt_on_lifull_rooms/gt_on_lifull', 'out_filename': '../data/results/gt_on_lifull_stats/inceptionv3_stats.npz'},
        ]

        device = 'cuda:0'
        dims = 2048
        batch_size = 100
        model = get_model(dims=dims, device=device)
        
        for rsi, result_set in enumerate(result_sets):

            room_basepath = result_set['room_basepath']
            out_filename = result_set['out_filename']

            print(f'result set [{rsi+1}/{len(result_sets)}]: {out_filename}')
        
            sample_names = get_room_sample_names(base_path=room_basepath)

            batch_size = 100
            batch_count = math.ceil(len(sample_names) / batch_size)
            activations = np.zeros((len(sample_names), dims))
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
                images = np.stack(images, axis=0)

                activations[samples_from:samples_to] = compute_activations_for_images(img_batch=images, model=model, device=device)

            mu, sigma = compute_statistics(activations=activations)
            os.makedirs(os.path.dirname(out_filename), exist_ok=True)
            np.savez(out_filename, mu=mu, sigma=sigma)


    if compute_stat_distances:

        stat_dist_sets = [
            # {'real_stat_filename': '../data/results/gt_on_rplan_stats/inceptionv3_stats.npz', 'fake_stat_filename': '../data/results/5_tuple_on_rplan_stats/temp_0.9_doors_0.9_walls_0.9/inceptionv3_stats.npz'},
            # {'real_stat_filename': '../data/results/gt_on_rplan_stats/inceptionv3_stats.npz', 'fake_stat_filename': '../data/results/5_tuple_on_rplan_stats/temp_0.9_doors_0.9_walls_1.0/inceptionv3_stats.npz'},
            # {'real_stat_filename': '../data/results/gt_on_rplan_stats/inceptionv3_stats.npz', 'fake_stat_filename': '../data/results/5_tuple_on_rplan_stats/temp_0.9_doors_1.0_walls_0.9/inceptionv3_stats.npz'},
            # {'real_stat_filename': '../data/results/gt_on_rplan_stats/inceptionv3_stats.npz', 'fake_stat_filename': '../data/results/5_tuple_on_rplan_stats/temp_0.9_doors_1.0_walls_1.0/inceptionv3_stats.npz'},
            # {'real_stat_filename': '../data/results/gt_on_rplan_stats/inceptionv3_stats.npz', 'fake_stat_filename': '../data/results/5_tuple_on_rplan_stats/temp_1.0_doors_0.9_walls_0.9/inceptionv3_stats.npz'},
            # {'real_stat_filename': '../data/results/gt_on_rplan_stats/inceptionv3_stats.npz', 'fake_stat_filename': '../data/results/5_tuple_on_rplan_stats/temp_1.0_doors_0.9_walls_1.0/inceptionv3_stats.npz'},
            # {'real_stat_filename': '../data/results/gt_on_rplan_stats/inceptionv3_stats.npz', 'fake_stat_filename': '../data/results/5_tuple_on_rplan_stats/temp_1.0_doors_1.0_walls_0.9/inceptionv3_stats.npz'},
            # {'real_stat_filename': '../data/results/gt_on_rplan_stats/inceptionv3_stats.npz', 'fake_stat_filename': '../data/results/5_tuple_on_rplan_stats/temp_1.0_doors_1.0_walls_1.0/inceptionv3_stats.npz'},

            # {'real_stat_filename': '../data/results/gt_on_lifull_stats/inceptionv3_stats.npz', 'fake_stat_filename': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_stats/temp_0.9_doors_0.9_walls_0.9/inceptionv3_stats.npz'},
            # {'real_stat_filename': '../data/results/gt_on_lifull_stats/inceptionv3_stats.npz', 'fake_stat_filename': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_stats/temp_0.9_doors_0.9_walls_1.0/inceptionv3_stats.npz'},
            # {'real_stat_filename': '../data/results/gt_on_lifull_stats/inceptionv3_stats.npz', 'fake_stat_filename': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_stats/temp_0.9_doors_1.0_walls_0.9/inceptionv3_stats.npz'},
            # {'real_stat_filename': '../data/results/gt_on_lifull_stats/inceptionv3_stats.npz', 'fake_stat_filename': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_stats/temp_0.9_doors_1.0_walls_1.0/inceptionv3_stats.npz'},
            # {'real_stat_filename': '../data/results/gt_on_lifull_stats/inceptionv3_stats.npz', 'fake_stat_filename': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_stats/temp_1.0_doors_0.9_walls_0.9/inceptionv3_stats.npz'},
            # {'real_stat_filename': '../data/results/gt_on_lifull_stats/inceptionv3_stats.npz', 'fake_stat_filename': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_stats/temp_1.0_doors_0.9_walls_1.0/inceptionv3_stats.npz'},
            # {'real_stat_filename': '../data/results/gt_on_lifull_stats/inceptionv3_stats.npz', 'fake_stat_filename': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_stats/temp_1.0_doors_1.0_walls_0.9/inceptionv3_stats.npz'},
            # {'real_stat_filename': '../data/results/gt_on_lifull_stats/inceptionv3_stats.npz', 'fake_stat_filename': '/home/guerrero/scratch_space/floorplan/results/5_tuple_on_lifull_stats/temp_1.0_doors_1.0_walls_1.0/inceptionv3_stats.npz'},

            # {'real_stat_filename': '../data/results/gt_on_rplan_stats/inceptionv3_stats.npz', 'fake_stat_filename': '../data/results/3_tuple_on_rplan_stats/nodes_0.9_0.9_doors_0.9_walls_0.9/inceptionv3_stats.npz'},
            # {'real_stat_filename': '../data/results/gt_on_lifull_stats/inceptionv3_stats.npz', 'fake_stat_filename': '../data/results/3_tuple_on_lifull_stats/nodes_0.9_0.9_doors_1.0_walls_1.0/inceptionv3_stats.npz'},
            # # {'real_stat_filename': '../data/results/gt_on_rplan_stats/inceptionv3_stats.npz', 'fake_stat_filename': '../data/results/3_tuple_on_rplan_stats/nodes_0.9_0.9_doors_0.9_walls_0.9_post_edges/inceptionv3_stats.npz'},

            # {'real_stat_filename': '../data/results/gt_on_rplan_stats/inceptionv3_stats.npz', 'fake_stat_filename': '../data/results/rplan_on_rplan_stats/inceptionv3_stats.npz'},
            {'real_stat_filename': '../data/results/gt_on_lifull_stats/inceptionv3_stats.npz', 'fake_stat_filename': '../data/results/rplan_on_lifull_stats/inceptionv3_stats.npz'},

            # {'real_stat_filename': '../data/results/gt_on_rplan_stats/inceptionv3_stats.npz', 'fake_stat_filename': '../data/results/stylegan_on_rplan_stats/inceptionv3_stats.npz'},
            # {'real_stat_filename': '../data/results/gt_on_lifull_stats/inceptionv3_stats.npz', 'fake_stat_filename': '../data/results/stylegan_on_lifull_stats/inceptionv3_stats.npz'},

            # {'real_stat_filename': '../data/results/gt_on_rplan_stats/inceptionv3_stats.npz', 'fake_stat_filename': '../data/results/graph2plan_on_rplan_stats/inceptionv3_stats.npz'},
            # {'real_stat_filename': '../data/results/gt_on_lifull_stats/inceptionv3_stats.npz', 'fake_stat_filename': '../data/results/graph2plan_on_lifull_stats/inceptionv3_stats.npz'},
        ]

        for rsi, stat_dist_set in enumerate(stat_dist_sets):
            real_stat_filename = stat_dist_set['real_stat_filename']
            fake_stat_filename = stat_dist_set['fake_stat_filename']
            out_filename = os.path.join(os.path.dirname(fake_stat_filename), 'fid.txt')

            print(f'stat distance set [{rsi+1}/{len(stat_dist_sets)}]: {out_filename}')

            with np.load(real_stat_filename) as npz_file:
                real_mu = npz_file['mu']
                real_sigma = npz_file['sigma']

            with np.load(fake_stat_filename) as npz_file:
                fake_mu = npz_file['mu']
                fake_sigma = npz_file['sigma']

            fid_value = calculate_frechet_distance(mu1=real_mu, sigma1=real_sigma, mu2=fake_mu, sigma2=fake_sigma)

            os.makedirs(os.path.dirname(out_filename), exist_ok=True)
            with open(out_filename, 'w') as f:
                f.write(f'{fid_value}\n')
