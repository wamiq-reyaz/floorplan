import os, sys
from multiprocessing import Pool
import numpy as np
from PIL import Image
from utils import rplan_map


def color_and_save(tuple_name):
    with open(tuple_name, 'rb') as fd:
        boxes = np.load(fd)
        boxes = boxes['arr_0']

    base_img = np.zeros((64, 64, 3), dtype=np.uint8)

    for box in boxes:
        id = box[0]
        if id >= rplan_map.shape[0]:
            print(id)
            continue
        x = box[1]
        y = box[2]
        w = box[3]
        h = box[4]

        color = np.around(rplan_map[id]*255)
        # print(color)
        base_img[y:y+h, x:x+w] = color

    base_file_name = os.path.basename(tuple_name)
    root_file_name = os.path.splitext(base_file_name)[0]
    save_file_name = os.path.join('samples', 'logged_0.8', 'rgb', root_file_name+'.png')

    img = Image.fromarray(base_img)
    img.save(save_file_name)


if __name__ == '__main__':
    from glob import glob
    all_tuples = glob('./samples/logged_0.8/*.npz')
    print(len(all_tuples))

    thread_pool = Pool(30)

    thread_pool.map(color_and_save, all_tuples)