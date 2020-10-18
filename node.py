import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import  Polygon
from matplotlib.collections import  PatchCollection
import numpy as np
import networkx as nx
import gurobipy as gp
from gurobipy import GRB
from scipy import ndimage
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from utils import show_with_grid, pairwise

class DimTuple(object):
    def __init__(self, first: float = 0, second: float = 0):
        self._data = [first, second]

    def __repr__(self):
        return tuple(self._data)

    def __getitem__(self, item: int):
        if not isinstance(item, (int)):
            raise ValueError('Only index 0 or 1 allowed')
        if item not in [0, 1]:
            raise ValueError('Only index 0 or 1 allowed')

        return self._data[item]


class AABB(object):
    def __init__(self, locs: DimTuple = DimTuple(), dims: DimTuple = DimTuple()):
        self._locs = locs
        self._dims = dims

    @classmethod
    def from_data(cls, xmin:float, ymin:float, h:float, w:float):
        locs = DimTuple(xmin, ymin)
        dims = DimTuple(w, h)
        return cls(locs, dims)

    def __repr__(self):
        return f"x_min: {self._locs[0]:0.3f}, x_max: {self._locs[0] + self._dims[0]:0.3f}," \
               f"y_min: {self._locs[1]:0.3f}, y_max: {self._locs[1] + self._dims[1]:0.3f}"

    def get_width(self) -> float:
        return self._dims[0]

    def get_height(self) -> float:
        return self._dims[1]

    def getx(self) -> float:
        return self._locs[0]

    def gety(self) -> float:
        return self._locs[1]

    def get_area(self) -> float:
        return self.get_width() * self.get_height()


class Node(object):
    def __init__(self, class_id:int = None, aabb:AABB = AABB()):
        self._class = class_id
        self._aabb = aabb

    @classmethod
    def from_data(cls, class_id:int = None, xmin:float = 0 , ymin:float = 0,
                  h:float = 0, w:float = 0):
        aabb = AABB.from_data(xmin, ymin, h, w)
        return cls(class_id=class_id, aabb=aabb)

    def __repr__(self):
        return f'Node: class_id: {self._class}, aabb: {self._aabb.__repr__()}'

    def set_class(self, class_id: int):
        self._class = class_id

    def get_class(self) -> int:
        return self._class

    def getx(self) -> float:
        return self._aabb.getx()

    def gety(self) -> float:
        return self._aabb.gety()

    def get_height(self) -> float:
        return self._aabb.get_height()

    def get_width(self) -> float:
        return self._aabb.get_width()

    def get_area(self) -> float:
        return self.get_width() * self.get_height()

class Floor(object):
    def __init__(self):
        self._rooms = []
        self.horiz_constraints = nx.DiGraph(name='horiz')
        self.vert_constraints = nx.DiGraph(name='vert')

        self._colormap = np.array(
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

        self._names = ['Exterior',
                       'Wall',
                       'Kitchen',
                       'Bedroom',
                       'Bathroom',
                       'Living Room',
                       'Office',
                       'Garage',
                       'Balcony',
                       'Hallway',
                       'Other Room']

    def __repr__(self):
        out = ''
        for ii, rr in enumerate(self._rooms):
            out += f'Room {ii}: ' + rr.__repr__() + '\n'

        return out

    def get_nrooms(self) -> int:
        return len(self._rooms)

    def add_horiz_constraints(self, edges) -> None:
        self.horiz_constraints.add_edges_from(edges)

    def add_vert_constraints(self, edges) -> None:
        self.vert_constraints.add_edges_from(edges)

    def add_room(self, room) -> None:
        self._add_node_to_graph()
        self._rooms.append(room)

    def _add_node_to_graph(self) -> None:
        idx = self.get_nrooms()
        self.horiz_constraints.add_node(idx, name=str(idx))
        self.vert_constraints.add_node(idx, name=str(idx))

    def get_width(self) -> float:
        width = 0
        for rr in self._rooms:
            width += rr.get_width()

        return width

    def get_height(self) -> float:
        height = 0
        for rr in self._rooms:
            height += rr.get_height()

        return height


    def draw(self, ax = None, both_labels=True, text=True, text_size=10):
        if ax is None:
            ax = plt.subplot(111)

        patches = []
        annots = []
        for ii, rr in enumerate(self._rooms):
            x = rr.getx() * 64
            y = rr.gety() * 64
            w = rr.get_width() * 64
            h = rr.get_height() * 64
            coords = [[x, y],
                      [x+w, y],
                      [x+w, y+h],
                      [x, y+h]]
            patches.append(
                Polygon(coords,
                       closed=True,
                       edgecolor=self._get_color(rr.get_class(), 1.0),
                       facecolor=self._get_color(rr.get_class(), 0.5),
                       linewidth=2
                       )
            )

            if text:
                x_middle = x + w * 0.5
                y_middle = y + h * 0.5
                label_string = '\textbf{' + str(ii) + '}'
                label_string += ', \textsc{' + self._names[rr.get_class()] + '}'

                color = 'white' if not rr.get_class() == 5 else 'black'
                text = plt.text(x_middle, y_middle,
                                 label_string.encode('unicode-escape').decode(),
                                {'color': color,
                                 'fontsize': text_size,
                                 'ha': 'center',
                                 'va': 'center',
                                 'bbox':{
                                        'boxstyle':'round',
                                        'fc': self._get_color(rr.get_class(), 0.8),
                                        'ec': self._get_color(rr.get_class(), 1.0),
                                        'pad':0.25
                                 }
                                 },
                                usetex=True,
                                # transform=ax.transAxes
                                )
                ax.add_artist(text)

        p = PatchCollection(patches, match_original=True)

        ax.add_collection(p)

        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')

        if both_labels:
            ax_h = ax.twiny()
            ax_v = ax.twinx()
            axes = [ax, ax_h, ax_v]
        else:
            axes = [ax]

        for curr_ax in axes:
            curr_ax.tick_params(axis='both', which='major', labelsize=16)
            curr_ax.tick_params(axis='both', which='minor', labelsize=10)

        return ax

    def _get_color(self, idx:int, alpha=1.0):
        return [*(self._colormap[idx].ravel()), alpha]





class LPSolver(object):
    PERIMETER = 0
    AREA = 1

    def __init__(self, floor:Floor):
        self._floor = floor
        self._n_vars = self._floor.get_nrooms()
        self._model = gp.Model('solver')
        self.xlocs = self._model.addMVar(shape=self._n_vars,  vtype=GRB.CONTINUOUS, name='xlocs')
        self.ylocs = self._model.addMVar(shape=self._n_vars, vtype=GRB.CONTINUOUS, name='ylocs')
        self.widths = self._model.addMVar(shape=self._n_vars, vtype=GRB.CONTINUOUS, name='widths')
        self.heights = self._model.addMVar(shape=self._n_vars, vtype=GRB.CONTINUOUS, name='heights')
        self.bbox_width = self._model.addMVar(shape=1, lb=0.0, vtype=GRB.CONTINUOUS, name='bbox_width')
        self.bbox_height = self._model.addMVar(shape=1, lb=0.0, vtype=GRB.CONTINUOUS, name='bbox_height')
        self.summer = np.ones((self._n_vars))
        self._min_sep = 0
        self.lines_align = False

    def __repr__(self):
        pass

    def same_line_constraints(self):
        self.lines_align = True

    def get_floor(self):
        return floor

    def solve(self, mode, iter=None):
        # self._model.setObjective(self.summer.T @ self.widths + self.summer.T @ self.heights, GRB.MINIMIZE)
        self._model.setObjective(self.bbox_width + self.bbox_height, GRB.MINIMIZE)

        self._read_graph()
        if iter is not None:
            self._model.setParam(GRB.Param.BarIterLimit, iter)
            self._model.setParam(GRB.Param.IterationLimit, iter)

        try:
            self._model.optimize()


        except gp.GurobiError as e:
            print('Error code ' + str(e.errno) + ": " + str(e))

        except AttributeError:
            print('Encountered an attribute error')

    def _set_floor_data(self):
        for ii, (x, y, w, h) in enumerate(
                        zip(self.xlocs.X, self.ylocs.X, self.widths.X, self.heights.X)):
            self._floor._rooms[ii]._aabb = AABB.from_data(x, y, h, w)

    def _build_constraints(self):
        pass

    def set_min_separation(self, sep:float):
        self._min_sep = sep

    def _read_graph(self):
        # horizontal constraints first
        for ii, e in enumerate(self._floor.horiz_constraints.edges):
            l = e[0]
            r = e[1]
            if self.lines_align:
                self._model.addConstr(self.xlocs[l] + self.widths[l] == self.xlocs[r], name=f'horiz_{ii}')
            else:
                self._model.addConstr(self.xlocs[l] + self.widths[l] <= self.xlocs[r] - self._min_sep, name=f'horiz_{ii}')



        for ii, e in enumerate(self._floor.vert_constraints.edges):
            b = e[0]
            t = e[1]
            if self.lines_align:
                self._model.addConstr(self.ylocs[b] + self.heights[b] == self.ylocs[t],  name=f'horiz_{ii}')
            else:
                self._model.addConstr(self.ylocs[b] + self.heights[b] <= self.ylocs[t] - self._min_sep, name=f'vert_{ii}')

        # constraints for the right/top

        for node in self._get_all_maximal(self._floor.horiz_constraints):
            # print(self.xlocs[node])
            # print(self.bbox_width)
            if self.lines_align:
                self._model.addConstr(self.xlocs[node] + self.widths[node] == self.bbox_width[0] , name=f'h_maximal_{node}')
            else:
                self._model.addConstr(self.xlocs[node] + self.widths[node] - self.bbox_width[0] + self._min_sep <= 0.0, name=f'h_maximal_{node}')

        for node in self._get_all_maximal(self._floor.vert_constraints):
            if self.lines_align:
                self._model.addConstr(self.ylocs[node] + self.heights[node] == self.bbox_height[0] , name=f'v_maximal_{node}')
            else:
                self._model.addConstr(self.ylocs[node] + self.heights[node] - self.bbox_height[0] + self._min_sep <= 0.0, name=f'v_maximal_{node}')

        # constraints for all

        for node in self._floor.horiz_constraints.nodes:
            self._model.addConstr(self.xlocs[node] >= self._min_sep, name=f'h_minimal_{node}')
            self._model.addConstr(self.ylocs[node] >= self._min_sep, name=f'v_minimal_{node}')



    def _get_all_maximal(self, graph):
        components = list(nx.weakly_connected_components(graph))

        maximal = []
        for cc in components:
            subgraph = graph.subgraph(cc)
            maximal.append(list(nx.topological_sort(subgraph))[-1])

        return maximal

    def _add_aspect_constraints(self):
        pass

    def _add_min_area_constrains(self, areas:list):
        if not len(areas) == self._n_vars:
            raise ValueError('The minimum areas should be same number as the number of rooms')

        for ii, ar in enumerate(areas):
            self._model.addConstr( self.widths[ii] @ self.heights[ii] >= ar , name=f'area_{ii}')

    def _add_width_constraints(self, widths:list, eps=0.1):
        if not len(widths) == self._n_vars:
            raise ValueError('The widths should be the same number as the number of rooms')

        was_list = False
        if isinstance(eps, list):
            was_list = True
            eps_list = eps.copy()
            if not len(eps) == self._n_vars:
                raise ValueError('The epsilons should have the number as rooms')

        for ii, ww in enumerate(widths):
            if was_list:
                eps = eps_list[ii]
            self._model.addConstr(self.widths[ii] >= widths[ii] * (1 - eps), name=f'width_min_{ii}')
            self._model.addConstr(self.widths[ii] <= widths[ii] * (1 + eps), name=f'width_max_{ii}')

    def _add_height_constraints(self, heights:list, eps=0.1):
        if not len(heights) == self._n_vars:
            raise ValueError('The widths should be the same number as the number of rooms')

        was_list = False
        if isinstance(eps, list):
            was_list = True
            eps_list = eps.copy()
            if not len(eps) == self._n_vars:
                raise ValueError('The epsilons should have the number as rooms')

        for ii, ww in enumerate(heights):
            if was_list:
                eps = eps_list[ii]
            self._model.addConstr(self.heights[ii] >= heights[ii] * (1 - eps), name=f'height_min_{ii}')
            self._model.addConstr(self.heights[ii] <= heights[ii] * (1 + eps), name=f'height_max_{ii}')

    def _add_xloc_constraints(self, xlocs:list, eps=0.1):
        if not len(xlocs) == self._n_vars:
            raise ValueError('The xlocs should be the same number as the number of rooms')

        was_list = False
        if isinstance(eps, list):
            was_list = True
            eps_list = eps.copy()
            if not len(eps) == self._n_vars:
                raise ValueError('The epsilons should have the number as rooms')

        for ii, ww in enumerate(xlocs):
            if was_list:
                eps = eps_list[ii]
            self._model.addConstr(self.xlocs[ii] >= xlocs[ii] * (1 - eps), name=f'width_min_{ii}')
            self._model.addConstr(self.xlocs[ii] <= xlocs[ii] * (1 + eps), name=f'width_max_{ii}')

    def _add_yloc_constraints(self, ylocs:list, eps=0.1):
        if not len(ylocs) == self._n_vars:
            raise ValueError('The widths should be the same number as the number of rooms')

        was_list = False
        if isinstance(eps, list):
            was_list = True
            eps_list = eps.copy()
            if not len(eps) == self._n_vars:
                raise ValueError('The epsilons should have the number as rooms')

        for ii, ww in enumerate(ylocs):
            if was_list:
                eps = eps_list[ii]
            self._model.addConstr(self.ylocs[ii] >= ylocs[ii] * (1 - eps), name=f'height_min_{ii}')
            self._model.addConstr(self.ylocs[ii] <= ylocs[ii] * (1 + eps), name=f'height_max_{ii}')


    def _add_aligment_constraints(self):
        pass

    def _add_symmetry_constraints(self):
        pass

    def _add_spacing_constraints(self):
        pass

    def is_solved(self):
        pass




class STNode(object):
    """ I don't know what it should contain
    """
    HORIZONTAL = True
    VERTICAL = not HORIZONTAL
    def __init__(self, aabb:AABB = None, children = [], direction=VERTICAL, idx=None):

        self.aabb = aabb
        self._children = children
        self.direction = direction
        self.idx = idx

    @property
    def children(self):
        return self._children

    @children.setter
    def children(self, value):
        self._children = value

    @property
    def xmin(self):
        return self.aabb.getx()

    @property
    def xmax(self):
        return self.aabb.get_width() + self.xmin

    @property
    def ymin(self):
        return self.aabb.gety()

    @property
    def ymax(self):
        return self.aabb.get_height() + self.ymin


    def add_child(self, ii, child_node):
        self.children = self.children + [child_node]

    def get_children(self):
        return self.children

    def make_terminal(self, idx):
        self.idx = idx
        self.children = None

    def is_terminal(self):
        return self.children is None

    def is_horiz_mergeable(self, other):

        if self.xmax == other.xmin: #x matches
            if self.ymin == other.ymin and self.ymax == other.ymax:
                return True

        return False

    def is_vert_mergeable(self, other):

        if self.ymax == other.ymin: #y matches
            if self.xmin == other.xmin and self.xmax == other.xmax:
                return True

        return False

    def is_class_equal(self, other):
        if self.idx is None or other.idx is None:
            raise ValueError('Only terminal boxes can be merged')

        if self.idx == other.idx:
            return True

        return False

    def get_merged(self, other):
        xmin = self.xmin
        ymin = self.ymin
        h = self.ymax - self.ymin
        w = other.xmax - self.xmin

        return STNode(
            aabb=AABB.from_data(xmin, ymin, h, w),
            idx=self.idx
        )

    def get_vmerged(self, other):
        xmin = self.xmin
        ymin = self.ymin
        h = other.ymax - self.ymin
        w = self.xmax - self.xmin

        return STNode(
            aabb=AABB.from_data(xmin, ymin, h, w),
            idx=self.idx
        )

    def is_hadj(self, other):
        if self.xmax == other.xmin:# or self.xmin == other.xmax:
            if max(self.ymin, other.ymin) <= min(self.ymax, other.ymax):
                return True 
            if max(other.ymin, self.ymin) <= min(other.ymax, self.ymax):
                return True

            # if self.ymax > other.ymin > self.ymin:
            #     return True

            # if self.ymax > other.ymin > self.ymin:
            #     return True

            # if self.ymax >= other.ymax and self.ymin < other.ymin:
            #     return True

            # if self.ymax > other.ymax and self.ymin <= other.ymin:
            #     return True

        return False

    def is_vadj(self, other):
        if self.ymax == other.ymin:# or self.ymin == other.ymax:
            if max(self.xmin, other.xmin) <= min(self.xmax, other.xmax):
                return True 
            if max(other.xmin, self.xmin) <= min(other.xmax, self.xmax):
                return True 

            # if self.xmax > other.xmin > self.xmin:
            #     return True

            # if self.xmax > other.xmin > self.xmin:
            #     return True

            # if self.xmax >= other.xmax and self.xmin < other.xmin:
            #     return True

            # if self.xmax > other.xmax and self.xmin <= other.xmin:
            #     return True

        return False

    def get_extent(self):
        xmin = self.aabb.getx()
        ymin = self.aabb.gety()

        xmax = self.aabb.get_width() + xmin
        ymax = self.aabb.get_height() + ymin

        return int(xmin), int(ymin), int(xmax), int(ymax)

    def get_area(self):
        return self.aabb.get_area()

    def get_width(self):
        return self.xmax - self.xmin

    def get_height(self):
        return self.ymax - self.ymin





class SplittingTree(object):
    def __init__(self, idx_img, cmap, grad_from='wall', door_img=None):
        if not isinstance(idx_img, (np.ndarray,)):
            raise ValueError('The image must be an np.array object')

        if not len(idx_img.shape) == 2:
            raise ValueError('The image must be indexed, not RGB')

        if not idx_img.dtype == np.uint8:
            try:
                idx_img = idx_img.astype(np.uint8)
            except Exception as e:
                raise e

        self.idx_img = idx_img
        self.door_img = door_img
        self.walls = idx_img == 1
        self.cmap = cmap
        self.img_height = self.idx_img.shape[0]
        self.img_width = self.idx_img.shape[1]

        self.head = STNode(
            aabb=AABB.from_data(0 ,0, self.img_height, self.img_width)
        )
        self.is_constructed = False
        self.gradh = None
        self.gradv = None
        self.leaves = None
        self.boxes = None
        self.horiz_adj = None
        self.vert_adj = None

        self.split_vert = None
        self.detect_wall = None
        self.grad_from = grad_from

    def _merge_small_boxes(self, cross_wall=True):
        self.boxes = []
        horiz = self.head.get_children()
        # print(len(horiz))
        # for ii, hbox in enumerate(horiz):
        #     print(ii, hbox.aabb)
        #
        #     for jj, vbox in enumerate(hbox.get_children()):
        #         print('\t\t', jj, vbox.aabb)

        horiz_lists = []
        for vbox in horiz:
            horiz_lists.append(vbox.get_children())

        # print('num horiz splits', len(horiz_lists))
        # print('total', len(self.leaves))
        # num_leaves = 0
        # for jj, vbox in enumerate(horiz_lists):
        #     print(f'{jj}: ', end='')
        #     print(len(vbox))
        #     num_leaves += len(vbox)
        #
        # print('total as per sum', num_leaves)

        for ii in range(len(horiz_lists)):
            added = self.boxes.copy()
            noadd_idx = []

            for jj, final_box in enumerate(self.boxes):
                for kk, box in enumerate(horiz_lists[ii]):
                    if final_box.is_horiz_mergeable(box) and final_box.is_class_equal(box):
                        if not cross_wall:
                            if self._is_hjoint_wall(final_box, box):
                                continue
                        added[jj] = final_box.get_merged(box)
                        noadd_idx.append(kk)
                        continue

            curr_full = horiz_lists[ii].copy()
            curr_selected = [vv for ii, vv in enumerate(curr_full) if ii not in noadd_idx]
            self.boxes = added + curr_selected


        # print(len(self.boxes))

    def _is_hjoint_wall(self, box1: STNode, box2:STNode):
        if not box1.xmax == box2.xmin:
            return False

        # slice original image
        wall_slice = self.idx_img[box1.ymin:box1.ymax, box1.xmax-1]
        idx = np.unique(wall_slice)

        wall_slice_right = self.idx_img[box1.ymin:box1.ymax, box2.xmin]
        idx_right = np.unique(wall_slice_right)

        if 1 in idx:
            if len(idx) == 1:
                return True

        if 1 in idx_right:
            if len(idx_right) == 1:
                return True

        return False


    def _is_vjoint_wall(self, box1, box2):
        if not box1.ymax == box2.ymin:
            return False

        # slice original image
        wall_slice = self.idx_img[box1.ymax-1, box1.xmin:box1.xmax]
        idx = np.unique(wall_slice)

        wall_slice_right = self.idx_img[box2.ymin, box1.xmin:box1.xmax]
        idx_right = np.unique(wall_slice_right)

        if 1 in idx:
            if len(idx) == 1:
                return True

        if 1 in idx_right:
            if len(idx_right) == 1:
                return True

        return False

    def _is_door_vert(self, box1, box2):
        if not box1.ymax == box2.ymin:
            return False

        # slice original image
        shared_xmin = max(box1.xmin, box2.xmin)
        shared_xmax = min(box1.xmax, box2.xmax)
        wall_slice = self.door_img[box1.ymax-1, shared_xmin:shared_xmax]
        idx = np.unique(wall_slice)

        wall_slice_right = self.door_img[box2.ymin, shared_xmin:shared_xmax]
        idx_right = np.unique(wall_slice_right)

        # print(wall_slice)

        if 1 in idx:
            if np.count_nonzero(wall_slice) > 1:
                return True

        if 1 in idx_right:
            if np.count_nonzero(wall_slice_right) > 1:
                return True

        return False

    def _is_door_horiz(self, box1: STNode, box2:STNode):
        if not box1.xmax == box2.xmin:
            return False

        # slice original image
        shared_ymin = max(box1.ymin, box2.ymin)
        shared_ymax = min(box1.ymax, box2.ymax)
        wall_slice = self.door_img[shared_ymin:shared_ymax, box1.xmax-1]
        idx = np.unique(wall_slice)

        wall_slice_right = self.door_img[shared_ymin:shared_ymax, box2.xmin]
        idx_right = np.unique(wall_slice_right)

        if 1 in idx:
            if np.count_nonzero(wall_slice) > 1:
                return True

        if 1 in idx_right:
            if np.count_nonzero(wall_slice_right) > 1:
                return True

        return False

    def _merge_vert_boxes(self, cross_wall=True):
        added = self.boxes.copy()
        # print(f'before merging vert, length added {len(added)}')
        noadd_idx = []

        for count in range(3):
            for ii, final_box in enumerate(self.boxes):
                for jj, box in enumerate(self.boxes):
                    if final_box == box:
                        continue
                    if final_box.is_vert_mergeable(box) and final_box.is_class_equal(box):
                        if not cross_wall:
                            if self._is_vjoint_wall(final_box, box):
                                continue
                        added[ii] = final_box.get_vmerged(box)
                        if ii not in noadd_idx:
                            noadd_idx.append(jj)
                        continue
            self.boxes = added

        self.boxes = [vv for ii, vv in enumerate(added) if ii not in noadd_idx]
        # print(f'after merging vert, len boxes {len(self.boxes)}')

    def create_tree(self):
        self._gen_gradh()
        self._gen_gradv()

        split_list = self._find_split_horiz(self.head)
        split_tuples = pairwise(split_list)

        direction = STNode.HORIZONTAL
        horiz_nodes = []
        for ii, (xmin, xmax) in enumerate(split_tuples):
            node = STNode(
                aabb=AABB.from_data(xmin, 0, self.img_height, xmax-xmin),
                direction=direction
            )
            self.head.add_child(0, node)
            horiz_nodes.append(node)

        # print(f'Num horiz nodes {len(horiz_nodes)}')
        # print(f'Num horiz nodes from self.head {len(self.head.get_children())}')
        # print(f'Head node {self.head}')


        direction = STNode.VERTICAL
        vert_nodes = []
        for ii, hnode in enumerate(horiz_nodes):
            xmin, ymin, xmax, ymax = hnode.get_extent()

            if self._is_uniform(xmin, xmax, ymin, ymax):
                # print(ii, xmin, ymin, xmax, ymax)
                node = STNode(
                    aabb=AABB.from_data(xmin, ymin, ymax - ymin, xmax - xmin),
                    direction=direction
                )
                hnode.add_child(0, node)
                # print('from horiz unif', ii,  len(self.head.get_children()))
                vert_nodes.append(node)
                continue

            # print(f'Num horiz nodes from self.head after vert {ii} {len(self.head.get_children())}')


            split_list = self._find_split_vert(hnode)
            split_tuples = pairwise(split_list)

            for jj, (ymin, ymax) in enumerate(split_tuples):
                node = STNode(
                    aabb=AABB.from_data(xmin, ymin, ymax-ymin, xmax-xmin),
                    direction=direction
                )
                hnode.add_child(0, node)
                vert_nodes.append(node)



        for ii, vnode in enumerate(vert_nodes):
            xmin, ymin, xmax, ymax = vnode.get_extent()
            # print(ii, xmin, ymin, xmax, ymax)
            # TODO uncomment
            if not self._is_uniform(xmin, xmax, ymin, ymax):
                raise ValueError
            vnode.make_terminal(self._get_idx(xmin, xmax, ymin, ymax))

        self.leaves = vert_nodes




        #TODO
        # find the horizontal splits
        # add horizontal children
        # the children have aabb given by lr horizontal, and tb = full image size
        # for all horizontal children
            #TODO
            # find vertical splits
            # if none, classify as end_node
            # add vertical children
            # the children have aabb given by lr of parent and tb vertical
            # for all children
                #TODO
                # check that the region is uniform
                # if uniform, mark as terminal

    def _is_uniform(self, xmin, xmax, ymin, ymax):
        img_slice = self.idx_img[ymin:ymax, xmin:xmax]

        if len(np.unique(img_slice)) == 1:
            # print('not removing wall')
            return True
        elif len(np.unique(img_slice)) == 2:
            # print('Removing wall')
            return True
        # elif len(np.unique(img_slice)) == 3:
        #     print(f'From unifrom {np.unique(img_slice)}')
        #     print(xmin, xmax, ymin, ymax)
        #     return True
        else:
            return False

    def _get_idx(self, xmin, xmax, ymin, ymax):
        if not self._is_uniform(xmin, xmax, ymin, ymax):
            raise KeyError('This slice is not uniform')
        img_slice = self.idx_img[ymin:ymax, xmin:xmax]
        # print(np.unique(img_slice))
        if len(np.unique(img_slice)) >= 2:
            idx = max(np.unique(img_slice))
            if idx == 1:
                idx = 0
            return idx


        return int(np.unique(img_slice))

    def find_horiz_adj(self):
        self.horiz_adj = nx.DiGraph()
        self.horiz_adj.add_nodes_from([(ii, {'idx':self.boxes[ii].idx}) for ii in range(len(self.boxes))])


        for source_idx, node in enumerate(self.boxes):
            for dest_idx, dnode in enumerate(self.boxes):
                if source_idx == dest_idx:
                    continue

                if node.is_hadj(dnode):# or dnode.is_hadj(node):
                    self.horiz_adj.add_edge(source_idx, dest_idx)


        return self.horiz_adj

    def find_vert_adj(self):
        self.vert_adj = nx.DiGraph()
        self.vert_adj.add_nodes_from([(ii, {'idx':self.boxes[ii].idx}) for ii in range(len(self.boxes))])

        for source_idx, node in enumerate(self.boxes):
            for dest_idx, dnode in enumerate(self.boxes):
                if source_idx == dest_idx:
                    continue

                if node.is_vadj(dnode):# or dnode.is_vadj(node):
                    self.vert_adj.add_edge(source_idx, dest_idx)


        return self.vert_adj

    def find_horiz_door(self):
        self.horiz_door = nx.DiGraph()
        self.horiz_door.add_nodes_from([(ii, {'idx':self.boxes[ii].idx}) for ii in range(len(self.boxes))])

        for source_idx, node in enumerate(self.boxes):
            for dest_idx, dnode in enumerate(self.boxes):
                if source_idx == dest_idx:
                    continue

                if self._is_door_horiz(node, dnode):  # or dnode.is_vadj(node):
                    self.horiz_door.add_edge(source_idx, dest_idx)

        return self.horiz_door


    def find_vert_door(self):
        self.vert_door = nx.DiGraph()
        self.vert_door.add_nodes_from([(ii, {'idx':self.boxes[ii].idx}) for ii in range(len(self.boxes))])

        for source_idx, node in enumerate(self.boxes):
            for dest_idx, dnode in enumerate(self.boxes):
                if source_idx == dest_idx:
                    continue

                if self._is_door_vert(node, dnode):  # or dnode.is_vadj(node):
                    self.vert_door.add_edge(source_idx, dest_idx)

        return self.vert_door


    def show_graphs(self):
        f, ax = plt.subplots(1, 3, dpi=160, figsize=(4, 4), sharex=False, sharey=False)

        ax[0] = show_with_grid(self.cmap[self.idx_img], ax[0])

        patches = []
        locs = {}
        colors = []
        ecolors = []

        for ii, bbox in enumerate(self.boxes):
            x = bbox.aabb.getx()
            y = bbox.aabb.gety()
            w = bbox.aabb.get_width()
            h = bbox.aabb.get_height()
            idx = bbox.idx
            coords = [[x, y],
                      [x+w, y],
                      [x+w, y+h],
                      [x, y+h]]

            # base_color
            # print(self.cmap[idx].tolist() + [0.8])

            patches.append(
                Polygon(coords,
                       closed=True,
                       linewidth=0.5,
                       facecolor= self._get_color(idx, 0.8),
                       edgecolor= (0,0,0, 0.5),
                       )
            )

            locs[ii] = (x+w/2.0, y+h/2.0)
            colors.append(self._get_color(idx, 1.0))
            ecolors.append((0.0, 0.0, 0.0, 0.8))


            # print(self.cmap[idx].tolist() + [0.8])
        p = PatchCollection(patches, match_original=True)
        p2 = PatchCollection(patches.copy(), match_original=True)


        ax[1].add_collection(p)
        ax[1].set_xlim((0, 64))
        ax[1].set_ylim((64, 0))
        ax[1].set_visible(True)
        ax[1].set_aspect('equal')

        nx.draw_networkx_nodes(self.horiz_adj,
                               pos=locs,
                               node_color=colors,
                               edgecolors=ecolors,
                               node_size=10,
                               linewidths=0.5,
                               ax=ax[1])
        nx.draw_networkx_labels(self.horiz_adj,
                               pos=locs,
                               font_size=8,
                               font_color='r',
                               ax=ax[1])
        nx.draw_networkx_edges(self.horiz_adj,
                               pos=locs,
                               node_size=0.1,
                               linewidths=0.5,
                               arrowstyle='->',
                               ax=ax[1])

        ax[2].add_collection(p2)
        ax[2].set_xlim((0, 64))
        ax[2].set_ylim((64, 0))
        ax[2].set_visible(True)
        ax[2].set_aspect('equal')

        nx.draw_networkx_nodes(self.vert_adj,
                               pos=locs,
                               node_color=colors,
                               edgecolors=ecolors,
                               node_size=10,
                               linewidths=0.5,
                               ax=ax[2])
        nx.draw_networkx_labels(self.vert_adj,
                                pos=locs,
                                font_size=8,
                                font_color='red',
                                ax=ax[2])
        nx.draw_networkx_edges(self.vert_adj,
                               pos=locs,
                               node_size=0.1,
                               linewidths=0.5,
                               arrowstyle='->',
                               ax=ax[2])



        return f, ax

    def _find_split_horiz(self, node:STNode):
        # return xmin, ymin, xmax, ymax
        xmin, ymin, xmax, ymax = node.get_extent()
        threshed = self.gradh.copy()
        threshed[threshed != 0] = 1.0

        nzlist = np.nonzero(np.sum(threshed, axis=0))[0].ravel()
        # print(nzlist)
        # print(type(nzlist))

        #
        splits = np.nonzero(np.sum(threshed, axis=0))[0].ravel()
        splits = list(splits)
        if self.grad_from == 'whole':
            del splits[0::2]

        if splits[0] != 0:
            splits.insert(0, 0)

        if splits[-1] != self.img_width:
            splits.append(self.img_width)
        # print(splits)
        return splits


    def _find_split_vert(self, node):
        xmin, ymin, xmax, ymax = node.get_extent()
        xmin = int(xmin)
        xmax = int(xmax)

        threshed = self.gradv.copy()
        if self.split_vert == 'own':
            threshed = threshed[:, xmin:xmax]
        threshed[threshed != 0] = 1.0

        splits = np.nonzero(np.sum(threshed, axis=1))[0].ravel()
        splits = list(splits)
        # print(splits)

        if self.detect_wall == 'line':
            propsective = []
            for ss in splits:
                elem = np.unique(self.idx_img[ss, xmin:xmax])

                if 1 in elem:
                    propsective.append(ss)

            splits = propsective
        else:
            pass

        if self.grad_from == 'whole':
            del splits[0::2]

        # print(splits)
        if splits[0] != 0:
            splits.insert(0, 0)

        if splits[-1] != self.img_height:
            splits.append(self.img_height)

        return splits

    def _gen_gradh(self):
        kernel = np.array([[1.0, -1]])
        if self.grad_from == 'wall':
            gradh = ndimage.correlate(self.walls.astype(np.float), kernel)
            self.gradh = gradh > 0

        elif self.grad_from == 'whole':
            gradh = ndimage.correlate(self.idx_img.astype(np.float), kernel)
            self.gradh = gradh


    def _gen_gradv(self):
        kernel = np.array([[1.0], [-1]])
        if self.grad_from == 'wall':
            gradv = ndimage.correlate(self.walls.astype(np.float), kernel)
            self.gradv = gradv > 0

        elif self.grad_from == 'whole':
            gradv = ndimage.correlate(self.idx_img.astype(np.float), kernel)
            self.gradv = gradv

    def show_grads(self):
        # ensure that gradv and gradh exist
        self._gen_gradh()
        self._gen_gradv()
        # print(np.unique(self.gradh))

        f, ax = plt.subplots(2, 2, dpi=160, figsize=(4,4), sharex=False, sharey=False)
        gradh =  self.gradh
        gradv =  self.gradv
        grad = gradh + gradv
        _ = show_with_grid(self.cmap[self.idx_img], ax[0, 0])
        _ = show_with_grid(gradh, ax[1, 0])
        _ = show_with_grid(gradv, ax[0, 1])
        # _ = show_with_grid(grad, ax[1, 1])

        # ax[1, 1].set_visible(False)

        return f, ax

    def box_artist(self):
        patches = []
        for bbox in self.boxes:
            x = bbox.aabb.getx()
            y = bbox.aabb.gety()
            w = bbox.aabb.get_width()
            h = bbox.aabb.get_height()
            idx = bbox.idx
            coords = [[x, y],
                      [x+w, y],
                      [x+w, y+h],
                      [x, y+h]]

            # base_color
            # print(self.cmap[idx].tolist() + [0.8])

            patches.append(
                Polygon(coords,
                       closed=True,
                       linewidth=0.5,
                       facecolor= self._get_color(idx, 0.8),
                       edgecolor= (0,0,0, 0.5),
                       )
            )
            # print(self.cmap[idx].tolist() + [0.8])
        p = PatchCollection(patches, match_original=True)

        return p

    def show_boxes(self, kind='all'):
        f, ax = self.show_grads()


        patches = []

        if kind == 'all':
            box_list = self.leaves

        elif kind == 'merged':
            box_list = self.boxes

        for bbox in box_list:
            x = bbox.aabb.getx()
            y = bbox.aabb.gety()
            w = bbox.aabb.get_width()
            h = bbox.aabb.get_height()
            idx = bbox.idx
            coords = [[x, y],
                      [x+w, y],
                      [x+w, y+h],
                      [x, y+h]]

            # base_color
            # print(self.cmap[idx].tolist() + [0.8])

            patches.append(
                Polygon(coords,
                       closed=True,
                       linewidth=0.5,
                       facecolor= self._get_color(idx, 0.8),
                       edgecolor= (0,0,0, 0.5),
                       )
            )
            # print(self.cmap[idx].tolist() + [0.8])
        p = PatchCollection(patches, match_original=True)

        ax[1, 1].add_collection(p)
        ax[1, 1].set_xlim((0, 64))
        ax[1, 1].set_ylim((64, 0))
        ax[1, 1].set_visible(True)
        ax[1, 1].set_aspect('equal')
        # show_with_grid(None, ax[1, 1], 64)

        return f, ax

    def _get_color(self, idx:int, alpha=1.0):
        return [*(self.cmap[idx].ravel()), alpha]

    def get_leaves(self):
        pass



class FloorPlanParser(object):
    def __init__(self, cmap, wall_idx,
                bg_idx, grad):
        
        self.cmap = cmap
        self.wall_idx = wall_idx
        self.bg_idx = bg_idx


    def parse(self, img):
        pass

    def _construct_tree(self, img):
        pass

    
    


if __name__ == '__main__':
    import random
    random.seed(42)
    from random import random as rand

    room1 = Node.from_data(7, rand(), rand(), rand(), rand())
    room2 = Node.from_data(6, rand(), rand(), rand(), rand())
    room3 = Node.from_data(2, rand(), rand(), rand(), rand())
    room4 = Node.from_data(3, rand(), rand(), rand(), rand())
    room5 = Node.from_data(4, rand(), rand(), rand(), rand())

    print(room1)
    floor = Floor()
    for rr in [room1, room2, room3, room4, room5]:
        floor.add_room(rr)

    print(floor)
    print('Number of rooms: ', floor.get_nrooms())
    print('Width: ', floor.get_width())
    print('Height: ', floor.get_height())

    #drawing the constraint graphs
    # ax = plt.subplot(121)
    # nx.draw(floor.horiz_constraints, ax=ax, with_labels=True)
    # ax = plt.subplot(122)
    # nx.draw(floor.vert_constraints, ax=ax, with_labels=True)
    # plt.show()

    floor.add_horiz_constraints([(0, 2), (1, 2), (2, 4), (3, 4)])
    floor.add_vert_constraints([(1, 0), (0, 3), (2, 3)])

    # ax = plt.subplot(121)
    # nx.draw(floor.horiz_constraints, ax=ax, with_labels=True)
    # ax = plt.subplot(122)
    # nx.draw(floor.vert_constraints, ax=ax, with_labels=True)
    # plt.show()

    # for e in floor.horiz_constraints.edges:
    #     print(e)
    #     print(e[0])

    solver = LPSolver(floor)
    solver._read_graph()
    solver.set_min_separation(0.01)
    solver._add_min_area_constrains([0.2, 0.1, 0.31, 0.3, 0.25])
    # print(solver._model)
    solver.solve(mode=None)

    print(solver.widths.X)
    print(solver.heights.X)
    print(solver.xlocs.X)
    print(solver.ylocs.X)

    solver._set_floor_data()
    ax = plt.subplot(111)
    floor.draw(ax=ax)

    plt.show()

    #
    # from matplotlib.patches import Polygon
    # from matplotlib.collections import PatchCollection
    # patches = []
    # for (w, h, x, y) in zip(solver.widths.X, solver.heights.X,
    #                         solver.xlocs.X, solver.ylocs.X):
    #     coords = [[x,y],
    #               [x+w, y],
    #               [x+w, y+h],
    #               [x, y+h]]
    #     patches.append(Polygon(coords, True))
    #
    # p = PatchCollection(patches)
    #
    # ax = plt.subplot(111)
    # colors = 100*np.random.rand(len(patches))
    # p.set_array(np.array(colors))
    # ax.add_collection(p)
    # plt.show()








