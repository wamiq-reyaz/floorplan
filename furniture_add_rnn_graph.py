import numpy as np

def get_rnn_graph(bboxes, r):
    
    furn_count = bboxes.shape[0]
    
    # get axis-aligned bounding boxes
    # orientations = bboxes[:, 4] * ((np.pi*2) / 32)
    # rotmats = np.array([
    #     [np.cos(orientations), -np.sin(orientations)],
    #     [np.sin(orientations),  np.cos(orientations)]]).transpose(2, 0, 1)
    # bbox_verts = np.concatenate([
    #     bboxes[:, [1,2]] + bboxes[:, [3,4]] * np.array([[-0.5, -0.5]]),
    #     bboxes[:, [1,2]] + bboxes[:, [3,4]] * np.array([[0.5, -0.5]]),
    #     bboxes[:, [1,2]] + bboxes[:, [3,4]] * np.array([[0.5, 0.5]]),
    #     bboxes[:, [1,2]] + bboxes[:, [3,4]] * np.array([[-0.5, 0.5]])
    #     ], axis=1)
    # bbox_verts = bbox_verts.reshape(furn_count, 4, 2)
    # for i in range(furn_count):
    #     bbox_verts[i] = np.matmul(bbox_verts[i]-bboxes[i, [1,2]], rotmats[i].transpose())+bboxes[i, [1,2]]
    # aabbs = np.concatenate([bbox_verts.min(axis=1), bbox_verts.max(axis=1)], axis=1)
    
    # boxes are axis-aligned boxes defined by center_x, center_y, w, h
    aabbs = np.concatenate([
        bboxes[:, [1,2]] + bboxes[:, [3,4]] * np.array([[-0.5, -0.5]]),
        bboxes[:, [1,2]] + bboxes[:, [3,4]] * np.array([[0.5, 0.5]])], axis=1)

    # add edges between aabbs that have a gap smaller then the given radius
    adj = np.zeros(shape=[furn_count, furn_count], dtype=np.bool)
    for i1 in range(furn_count-1):
        for i2 in range(i1+1, furn_count):
            gap_dist = np.sqrt((np.maximum(0, np.maximum(aabbs[i1, :2] - aabbs[i2, 2:], aabbs[i2, :2] - aabbs[i1, 2:]))**2).sum())
            if gap_dist <= r:
                adj[i1, i2] = True

    neighbor_edges = np.hstack([x.reshape(-1, 1) for x in np.nonzero(adj)])

    return neighbor_edges

if __name__ == '__main__':
    import math
    from furniture_io import get_sample_names, load_furniture, save_furniture
    from tqdm import tqdm

    result_sets = [
        {'in_path': '../data/results/furniture/6_tuple_furn/6_tuple', 'out_path': '../data/results/furniture/6_tuple_rnngraph/6_tuple', 'r': 6},
        {'in_path': '../data/results/furniture/stylegan_furn/stylegan', 'out_path': '../data/results/furniture/stylegan_rnngraph/stylegan', 'r': 6},
        {'in_path': '/home/guerrero/scratch_space/floorplan/gt_furn/gt', 'out_path': '/home/guerrero/scratch_space/floorplan/gt_rnngraph/gt', 'r': 6},
    ]
    
    for rsi, result_set in enumerate(result_sets):

        in_path = result_set['in_path']
        out_path = result_set['out_path']
        r = result_set['r']

        print(f'result set [{rsi+1}/{len(result_sets)}]')
    
        # read the boxes and edges of all floor plans in the input directory
        sample_names = get_sample_names(base_path=in_path)

        batch_size = 100
        batch_count = math.ceil(len(sample_names) / batch_size)
        for batch_idx in tqdm(range(batch_count)):
            samples_from = batch_size*batch_idx
            samples_to = min(batch_size*(batch_idx+1), len(sample_names))
            batch_sample_names = sample_names[samples_from:samples_to]

            _, furn_bboxes, furn_neighbor_edges, furn_masks, room_bboxes, room_masks = load_furniture(
                base_path=in_path, sample_names=batch_sample_names)

            furn_neighbor_edges = []
            for si, sample_name in enumerate(batch_sample_names):
                neighbor_edges = get_rnn_graph(bboxes=furn_bboxes[si], r=r)
                furn_neighbor_edges.append(neighbor_edges)
            
            save_furniture(
                base_path=out_path,
                sample_names=batch_sample_names,
                furn_bboxes=furn_bboxes,
                furn_neighbor_edges=furn_neighbor_edges,
                furn_masks=furn_masks,
                room_bboxes=room_bboxes,
                room_masks=room_masks,
                append=batch_idx > 0)
