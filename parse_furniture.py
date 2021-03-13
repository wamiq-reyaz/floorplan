import os
import sys
import numpy as np
from tqdm import tqdm
from torch import  multiprocessing as mp

class Callback:
    def __init__(self, length):
        self.bar = tqdm(total=length, leave=True)
        self.output = []

    def update(self, ret):
        # print('callback')
        # self.output.append(1)
        self.bar.update(1)

    def aa(self, dd):
        print('asdfasdf')

    def close(self):
        self.bar.close()


# def aa(ret):
#     print('callback')



class FurnitureParser:
    def __init__(self,
                 file_list,
                 root,
                 subset=None
                 ):
        self.file_list = file_list
        self.root = root
        self.files = self._get_full_path()
        if subset is not None:
            self.files = self.files[subset[0]:subset[1]]
        self.walkable_idx = 27
        self.door_idx = 2



    def _get_full_path(self):
        with open(self.file_list, 'r') as fd:
            lines = fd.readlines()

        full_paths = []
        for ll in lines:
            full_path = os.path.join(self.root, ll.rstrip('\n') + '_graph.npy')
            full_paths.append(full_path)

        return full_paths

    def parse(self, idx):
        if not os.path.exists(self.files[idx]):
            new_path = os.path.join(self.root, 'af64a47387330dacf4638c0d99365c4b_0_graph.npy')
            print('fuck')
            with open(new_path, 'rb') as fd:
                full_mat = np.load(fd)
        else:
            with open(self.files[idx], 'rb') as fd:
                full_mat = np.load(fd)


        # print(full_mat[:3, :])
        # get room first
        room = full_mat[0, :]

        # drop first row, 6th col
        compact_mat = np.delete(full_mat, 0, axis=0)
        compact_mat = np.delete(compact_mat, 6, axis=1)

        # delete duplicates
        only_furn = compact_mat[:, :6]
        unq, return_idx = np.unique(only_furn, axis=0, return_index=True)
        ofsetted_idx = list((6+return_idx).ravel())

        compact_mat = compact_mat[return_idx, :]
        compact_mat = compact_mat[:, [0,1,2,3,4,5] +ofsetted_idx]
        # print(compact_mat.shape)

        # delete nodes that are merely walkable
        furn_idx = list(compact_mat[:, 0].ravel())

        wk_node_idx = [ii for ii, idx in enumerate(furn_idx) if idx == self.walkable_idx]
        wk_node_idx2 = [ii+6 for ii, idx in enumerate(furn_idx) if idx == self.walkable_idx]

        compact_mat = np.delete(compact_mat, wk_node_idx, axis=0)
        compact_mat = np.delete(compact_mat, wk_node_idx2, axis=1)

        # make centeres positive
        x_offset = room[3] //2
        y_offset = room[4] //2


        compact_mat[:, 1] += x_offset
        compact_mat[:, 2] += y_offset

        
        # # move doors to the front
        # furn_idx = list(compact_mat[:, 0].ravel()) # after delete_rows, again compute indices

        # door_node_idx = [ii for ii, idx in enumerate(furn_idx) if idx == self.door_idx]

        # doors = compact_mat
        # num_doors = len(door_node_idx)
        # swap_idx = list(range(num_doors))
        # # print(door_node_idx)

        # swapper_tgt = np.array([*swap_idx, *door_node_idx], dtype=np.int)
        # swapper_src = np.array([*door_node_idx, *swap_idx], dtype=np.int)

        # # print(swapper_tgt, swapper_src)
        # # print(compact_mat)
        # compact_mat[swapper_tgt, :] = compact_mat[swapper_src, :]
        # compact_mat[:, swapper_tgt+6] = compact_mat[:, swapper_src+6]
        # # print(compact_mat)



        # for the rooms, use raster order firstx then y
        #TODO 
        num_doors = 0
        raster_idx = 1000 * compact_mat[num_doors:, 1] + compact_mat[num_doors:, 2]
        sorted_idx = np.argsort(raster_idx)

        # swap rows first and then columns
        compact_mat[num_doors:, :] = compact_mat[num_doors+sorted_idx, :]
        compact_mat[:, num_doors+6:] = compact_mat[:, 6+num_doors+sorted_idx] # notice that the adj is offset


        adj_mat = compact_mat[:, 6:]
        nodes = compact_mat[:, :6]

        width_edg = adj_mat & 8
        height_edg = adj_mat & 16
        orient_edg = adj_mat & 32
        adj_edg = adj_mat & 64

        wh_shape = width_edg.shape == height_edg.shape
        oa_shape = orient_edg.shape == adj_edg.shape

        if not (wh_shape and oa_shape):
            raise ValueError('WTF')





        # return room, full_mat, compact_mat

        return room, nodes, width_edg, height_edg, orient_edg, adj_edg


def save(subset):
    fp = FurnitureParser(file_list='../furniture_018/all.txt',
                         root='../furniture_018/',
                         subset=subset)

    num_files = len(fp.files)

    for ii in range(num_files):
        room, full, compact = fp.parse(ii)

        base_name = os.path.splitext(fp.files[ii])[0]
        base_name = base_name.replace('_graph', '')
        name = base_name + '_xywh.npz'

        np.savez(name, room=room, furniture=compact)

    return 1

def save_edg(subset):
    fp = FurnitureParser(file_list='../furniture_018/all.txt',
                         root='../furniture_018/',
                         subset=subset)

    num_files = len(fp.files)

    for ii in range(num_files):
        r, n, w, h, o, a = fp.parse(ii)
        base_name = os.path.splitext(fp.files[ii])[0]
        base_name = base_name.replace('_graph', '')
        name = base_name + '_edges.npz'

        np.savez(name, r=r, n=n, w=w, h=h, o=o, a=a)

    # print('uuu')


    return 1



if __name__ == '__main__':

    file = '../furniture_018/all.txt'
    with open(file, 'r') as fd:
        file_names = fd.readlines()

    num_files = len(file_names)
    BATCH_SIZE = 1

    subsets = []
    for ii in range(2*num_files):
        lower = ii*BATCH_SIZE
        upper = min(BATCH_SIZE*(ii+1), num_files)
        subsets.append((lower, upper))

        if upper >= num_files:
            break



    print(len(subsets))

    
    from time import sleep
    with mp.Manager() as manager:
        with mp.Pool(15) as pool:
            cb = Callback(length=num_files)
            results = []
            for ss in subsets:
                rval = pool.apply_async(save_edg, args=(ss,), callback=cb.update)
                results.append(rval)

            for r in results:
                r.wait()
                r.get()

    print(cb.output)

    # rval.wait()
    # pool.close()
    # pool.join()
    # print('done')


    # print(rvals.get())

    # for ii in tqdm(range(num_files), total=num_files):
    #     room, full, compact = fp.parse(ii)
    #
    #     base_name = os.path.splitext(fp.files[ii])[0]
    #     base_name = base_name.replace('_graph', '')
    #     name = base_name + '_xywh.npz'
    #     # print(name)
    #     # name = '_xywh.npz'
    #
    #     np.savez(name, room=room, furniture=compact)
    #     # sys.exit()














