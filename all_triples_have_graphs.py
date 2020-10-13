import os
from glob import glob
import numpy as np
from tqdm import tqdm


if __name__ == '__main__':
    ROOT_DIR = './samples/triples_0.8'
    verts = glob(os.path.join(ROOT_DIR, '*.npz'))


    for vv in tqdm(verts):
        base_name = os.path.basename(vv)
        root_name = os.path.splitext(base_name)[0]

        horiz_file = os.path.join(ROOT_DIR, 'edges', 'h', root_name + '.pkl')
        vert_file = os.path.join(ROOT_DIR, 'edges', 'v', root_name + '.pkl')

        # print(horiz_file)
        assert os.path.exists(vv)
        assert os.path.exists(horiz_file)
        assert os.path.exists(vert_file)