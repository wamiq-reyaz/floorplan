import os, sys
from multiprocessing import Pool
import numpy as np
from PIL import Image
from utils import rplan_map


def color_and_save(tuple_name):
    with open(tuple_name, 'rb') as fd:
        boxes = np.load(fd)
        try:
            boxes = boxes['arr_0']
        except IndexError:
            pass

    base_img = np.ones((64, 64, 3), dtype=np.uint8) * 255

    for box in boxes:
        id = int(box[0])
        if id >= rplan_map.shape[0]:
            print(id)
            continue
        x = int(box[1])
        y = int(box[2])
        w = int(box[3])
        h = int(box[4])
        print(id)

        color = np.around(rplan_map[id]*255)
        # print(color)
        base_img[y:y+h, x:x+w] = color

    base_file_name = os.path.basename(tuple_name)
    root_file_name = os.path.splitext(base_file_name)[0]
    save_file_name = os.path.join('samples', 'triples_0.5', 'rgb', root_file_name+'.png')

    img = Image.fromarray(base_img)
    img.save(save_file_name)


if __name__ == '__main__':
    from glob import glob
    all_tuples = glob('/mnt/ibex/Projects/floorplan/samples/triples_0.5/nodes_0.5_0.5/*.npz')
    print(len(all_tuples))

    thread_pool = Pool(30)

    thread_pool.map(color_and_save, all_tuples)