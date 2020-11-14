from rplan import RrplanDoors
from torch.utils.data import DataLoader
from tqdm import tqdm
import  sys
import torch

if __name__ == '__main__':
    dset = RrplanDoors(root_dir='/mnt/iscratch/datasets/rplan_ddg_var',
                     split='train',
                     seq_len=120,
                     edg_len=220,
                     vocab_size=65,
                     dims=5,
                     doors='all',
                     ver2=True
                  )

    dloader = DataLoader(dset, num_workers=10, batch_size=64)

    with open('delete.txt', 'w') as fd:
        for ii, ss in tqdm(enumerate(dloader)):
            flat_list = ss['flat_list']
            all_names = ss['name']

            num_doors = flat_list.shape[0]
            print(num_doors)
            for jj in range(num_doors):
                if flat_list[jj] < 15:
                    print(all_names[jj], flat_list[jj], file=fd)

            # sys.exit()
    # with open('doors2.txt', 'w') as fd:
    #     for ii, ss in tqdm(enumerate(dloader)):
    #         node_idx = ss['nodes_exist']
    #         doors_idx = ss['doors_exist']
    #
    #         all_names = ss['name']
    #
    #         for jj, nn in enumerate(all_names):
    #             print(nn, node_idx[jj], doors_idx[jj], file=fd)



