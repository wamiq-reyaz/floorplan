import os, sys
from multiprocessing import Pool
import numpy as np
from PIL import Image, ImageDraw
from utils import rplan_map, furniture_colors_map, furniture_room_map
from scipy.ndimage import binary_dilation, binary_erosion

def color_and_save(tuple_name):
    with open(tuple_name, 'rb') as fd:
        boxes = np.load(fd)
        try:
            boxes = boxes['arr_0']
        except IndexError:
            pass



    multiplier = 4
    furn_multiplier = 2
    base_img = np.zeros((64 * multiplier, 64 * multiplier, 4), dtype=np.uint8)
    im = Image.fromarray(base_img, mode='RGBA')
    draw = ImageDraw.Draw(im, 'RGBA')

    boxes[:, [1,2,3,4]] = boxes[:, [1 ,2 ,3 ,4]] * furn_multiplier
    room = boxes[0, :]
    id = room[0]
    if id < 4 :
        id =4

    if id == 9:
        id = 6
    if id not in furniture_room_map.keys():
        return 
    roomx = room[3]
    roomy = room[4]
    
    # print(id)

    outline_color = np.around(np.asarray(furniture_room_map[id])*255).astype(np.int)

    xmin = (64*multiplier) - roomx
    xmin = xmin // 2
    xmax = xmin + roomx

    ymin = (64*multiplier) - roomy
    ymin = ymin //2
    ymax = ymin + roomy

    polygon = [(xmin, ymin), (xmin+roomx, ymin), (xmin+roomx, ymin+roomy), (xmin, ymin+roomy)]

    draw.polygon(polygon, outline=(*outline_color, 255), fill=None)

    boxes = boxes[1:, :]

    unq, return_idx = np.unique(boxes, axis=0, return_index=True)
    boxes = boxes[return_idx, :]

    raster_idx = 1000 * boxes[:, 1] + boxes[:, 2]
    sorted_idx = np.argsort(raster_idx)

    boxes = boxes[sorted_idx, :]

    for ii, box in enumerate(boxes):
        # print(box)
        id = int(box[0])
        if id not in furniture_colors_map.keys() :
            print(id)
            continue
        x = int(box[1]) + xmin
        y = int(box[2]) + ymin
        if len(box) == 3:
            w = 1*multiplier
            h = 1*multiplier
        else:
            w = int(box[3])
            h = int(box[4])

        x = x - w//2
        y = y - h//2
        # print(id)

        color = np.around(furniture_colors_map[id]).astype(np.int)

        polygon = [(x,y), (x+w, y), (x+w, y+h), (x, y+h)]
        draw.polygon(polygon, outline=(*color, 255), fill=(*color, 127))
        draw.text((xmin + int(box[1]), ymin+int(box[2])), str(ii+1))

    base_file_name = os.path.basename(tuple_name)
    root_file_name = os.path.splitext(base_file_name)[0]
    save_file_name = os.path.join('floorplan_suppl_kitchen_im', root_file_name+'.png')

    # img = Image.fromarray(base_img)
    im.save(save_file_name)

def open_npz(fname):
    with open(fname, 'rb') as fd:
        boxes = np.load(fd)
        try:
            boxes = boxes['arr_0']
        except IndexError:
            pass

    return boxes
def reshape_box_list(boxes, exterior=False):
    if exterior:
        boxes = boxes[1:] - 1

    stop_idx = np.argmax(boxes)
    try:
        boxes = boxes[:stop_idx]
        boxes = boxes.reshape((-1, 5))
    except:
        return None

    return boxes

def fill_boxes(boxes, alpha=255):
    base_img = np.ones((64, 64, 4), dtype=np.uint8) * 255

    is_valid = True
    for box in boxes:
        # print(box)
        id = int(box[0])
        if id >= rplan_map.shape[0]:
            # print(id)
            is_valid = False
            continue
        x = int(box[1])
        y = int(box[2])
        w = int(box[3])
        h = int(box[4])
        # print(id)

        color = np.around(rplan_map[id] * 255)
        color = np.hstack((color, [alpha]))
        # print(color)
        base_img[y:y + h, x:x + w] = color

    return base_img if is_valid else None

def draw_boundary(boxes, image):
    bound_img = np.zeros((64, 64))
    struct = np.ones((3,3))

    for box in boxes:
        x = int(box[1])
        y = int(box[2])
        w = int(box[3])
        h = int(box[4])
        print(box)
        bound_img[y:y + h, x:x + w] = 1

    bound = binary_erosion(bound_img, structure=struct, border_value=1)
    bound = bound - bound_img

    img_to_choose = np.zeros((64, 64, 4), dtype=np.uint8)
    img_to_choose[..., 3] = 127

    new_img = np.where(bound[:, :, None], img_to_choose, image)

    return new_img, bound

def save_img(img, fname, suffix='', is_float=False):
    base_file_name = os.path.basename(fname)
    root_file_name = os.path.splitext(base_file_name)[0]
    save_file_name = os.path.join(root_file_name + suffix + '.png')

    if is_float:
        img = img.astype(np.float).squeeze()
        # print(np.sum(img > 0))
    img = Image.fromarray(img)
    if is_float:
        img = img.convert('RGB')
    img.save(save_file_name)

def color_together(tuple_name):
    int_boxes = open_npz(tuple_name[0])
    ext_boxes = open_npz(tuple_name[1])

    int_boxes = reshape_box_list(int_boxes)
    ext_boxes = reshape_box_list(ext_boxes, exterior=True)
    if int_boxes is None or ext_boxes is None:
        print('fuck')
        return

    img = fill_boxes(int_boxes, alpha=200)
    img_bound, bound = draw_boundary(ext_boxes, img)

    save_img(img, fname=tuple_name[0], suffix='_bound')
    save_img(img_bound, fname=tuple_name[0], suffix='_only_bound', is_float=False)


if __name__ == '__main__':
    from glob import glob
    from natsort import natsorted
    # all_tuples = glob('/mnt/ibex/Projects/floorplan/samples/triples_0.5/nodes_0.5_0.5/*.npz')
    all_tuples = natsorted(glob('./floorplan_suppl_kitchen/*.npz'))
    # all_ext = natsorted(glob('./rplan3/exterior*.npz'))

    # together = list(zip(all_tuples, all_ext))

    # print(len(all_tuples))

    # print(all_tuples)
    thread_pool = Pool(16)


    # img, bound = color_together(together[0])
    #
    #
    # print(img.shape, img.dtype)
    # print(bound.shape, bound.dtype)
    #
    # import matplotlib.pyplot as plt
    # print(bound)
    # plt.imshow(img)
    # plt.show()
    thread_pool.map(color_and_save, all_tuples)
    # thread_pool.map(color_and_save_exterior, all_ext)
    # thread_pool.map(color_together, together)