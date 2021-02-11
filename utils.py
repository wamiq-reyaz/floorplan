from itertools import tee
from math import ceil
import numpy as np
import torch
import torch.nn.functional as F
import socket

rplan_map = np.array(
    [[0.0, 0.0, 0.0],
     [0.600000, 0.600000, 0.600000],
     [0.301961, 0.686275, 0.290196],
     [0.596078, 0.305882, 0.639216],
     [1.000000, 0.498039, 0.000000],
     [1.000000, 1.000000, 0.200000],
     [0.650980, 0.337255, 0.156863],
     [0.000000, 1.000000, 1.000000],
     [1.000000, 0.960784, 1.000000],
     [0.309804, 0.305882, 0.317647]])

furniture_colors_map = {1: (0,0,255),
                        2: (255,0,0),
                        # 3 is wall...
                        4: (232,186,215),
                        5: (209,84,163),
                        6: (94,19,67),
                        7: (19,24,48),
                        8: (94,72,0),
                        9: (0,45,48),
                        10: (159,84,209),
                        11: (125,139,209),
                        12: (108,94,117),
                        13: (94,116,117),
                        14: (186,112,112),
                        15: (102,245,255),
                        16: (255,243,150),
                        17: (117,112,94),
                        18: (119,255,0),
                        19: (146,186,112),
                        20: (0,71,21),
                        21: (37,186,82),
                        22: (232,0,0),
                        23: (0,109,117),
                        24: (140,0,0),
                        25: (48,0,0),
                        26: (255,102,102)
                        }

furniture_room_map = {4: (0.301961,0.686275,0.290196),
                      5: (0.596078,0.305882,0.639216),
                      6: (1.000000,0.498039,0.000000),
                      7: (1.000000,1.000000,0.200000),
                      8: (0.650980,0.337255,0.156863),
                     10: (0.000000,0.262745,0.003922),
                     11: (0.000000,1.000000,1.000000),
                     12: (0.372549,0.000000,0.160784),
                     13: (0.309804,0.305882,0.317647)
                      }

RPLAN_DOOR_COLOR = np.array((228, 26, 28))
RPLAN_DOOR_COLOR = tuple(RPLAN_DOOR_COLOR.ravel())


def colorize_rplan(preds):
    preds = torch.argmax(preds, dim=1).cpu().numpy()
    rgb_preds = rplan_map[preds]
    return rgb_preds


def make_rgb_indices(img, cmap):
    h, w, c = img.shape
    mask = np.zeros((h, w))
    nrows = cmap.shape[0]
    colors = np.around(cmap * 255)
    for ii in range(nrows):
        color = tuple(colors[ii, :].ravel())
        hit = (img == color).all(2)
        mask[hit] = ii
    return mask

def make_door_indices(img, door_color=RPLAN_DOOR_COLOR):
    h, w, c = img.shape
    mask = np.zeros((h, w))
    hit = (img == door_color).all(2)
    mask[hit] = 1

    return mask

def make_rgb_indices_rounding(img, cmap):
    h, w, c = img.shape
    mask = np.zeros((h, w))
    nrows = cmap.shape[0]
    colors = np.around(cmap * 255)
    for ii in range(nrows):
        color = tuple(colors[ii, :].ravel())
        diff = img - color
        diff_sqr = diff * diff
        diff_sum_sqr = diff_sqr.sum(2)
        hit = diff_sum_sqr < 5
        mask[hit] = ii
    return mask


def show_with_grid(im, ax, size=(64, 64)):
    if isinstance(size, (int)):
        try:
            size = (size, size)
        except Exception as e:
            raise e

    if not isinstance(size, (tuple)):
        raise ValueError('size must be a tuple of ints, or an int')

    print(size)

    round_tens = ceil(size[0] / 10.0)

    if im is not None:
        ax.imshow(im, interpolation='nearest', extent=(0, *size, 0))

    major_ticks = [10 * ii for ii in range(round_tens)]
    _ = ax.set_xticks(major_ticks, minor=False)
    _ = ax.set_yticks(major_ticks, minor=False)

    _ = ax.set_xticks(np.arange(size[0]), minor=True)
    _ = ax.set_yticks(np.arange(size[0]), minor=True)

    ax.grid(which='major', alpha=0.8)
    ax.grid(which='minor', alpha=0.5)

    _ = ax.set_xticklabels([str(10 * ii) for ii in range(round_tens)])
    _ = ax.set_yticklabels([str(10 * ii) for ii in range(round_tens)])

    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')

    return ax


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def on_local():
    if socket.gethostname() == 'PC-KW-60046':
        return True
def parse_wall_or_door_seq(seq):
    curr_node = 0
    edge_list = []

    seq_biased = seq - 2

    # drop all -1
    seq_as_list = list(seq_biased.ravel())
    seq_dropped = [s for s in seq_as_list if s != -1]
    # print(seq_dropped)

    seq_dropped_np = np.array(seq_dropped, dtype=np.int)
    stop_idx = np.argmin(seq_dropped_np)
    # print(stop_idx)
    seq_dropped_np = seq_dropped_np[:stop_idx]

    seq_final = list(seq_dropped_np.ravel())
    len_seq = len(seq_dropped_np)

    if len_seq % 2 != 0:
        print('yikes! this sample is bad')
        return 0

    seq_dropped_np = seq_dropped_np.reshape((-1, 2))

    num_edges = seq_dropped_np.shape[0]

    for ii in range(num_edges):
        curr_edg = tuple(list(seq_dropped_np[ii, :].ravel()))
        edge_list.append(curr_edg)

    return edge_list

def parse_edge_seq(seq):
    curr_node = 0
    edge_list = []

    seq_biased = seq - 2
    for elem in seq_biased:
        if elem == -1:
            curr_node += 1
            continue
        elif elem == -2:
            break
        else:
            edge_list.append((curr_node, elem))

    return edge_list

def parse_vert_seq(seq):
    if isinstance(seq, torch.Tensor):
        try:
            seq = seq.cpu().numpy()
        except:
            seq = seq.numpy()

    # print(seq)
    # first slice the head off
    seq = seq[2:, :]

    #find the max along any axis of id, x, y
    stop_token_idx = np.argmax(seq[:, 0])


    boxes = seq[:stop_token_idx, :] - 1

    return boxes

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits