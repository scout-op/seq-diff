import numpy as np
import cv2
import networkx as nx
import rtree
import scipy.ndimage
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString, Point
import warnings
import sys

# --- Helper Functions from CGNet's topo_metrics.py and geotopo.py ---

def f1_score(precision, recall):
    """
    Calculate F1 score given precision and recall.
    """
    if precision == 0 or recall == 0:
        return 0
    return 2 * ((precision * recall) / (precision + recall))

def calc_sda(graph_gt, graph_pred, threshold=1):
    """
    Calculates the split detection accuracy (SDA) metric for a pair of graphs and a given threshold.
    """
    split_point_positions_gt = []
    split_point_positions_pred = []

    for n in graph_gt.nodes():
        if graph_gt.out_degree(n) >= 2 or graph_gt.in_degree(n) >= 2:
            split_point_positions_gt.append(graph_gt.nodes[n]['pos'])

    for n in graph_pred.nodes():
        if graph_pred.out_degree(n) >= 2 or graph_pred.in_degree(n) >= 2: 
            split_point_positions_pred.append(graph_pred.nodes[n]['pos'])

    if len(split_point_positions_gt) == 0:
        return np.nan

    if len(split_point_positions_pred) == 0:
        return 0.0

    # build up cost matrix between gt and pred split points
    split_point_positions_gt = np.array(split_point_positions_gt)
    split_point_positions_pred = np.array(split_point_positions_pred)

    cost_matrix = np.linalg.norm(split_point_positions_gt[:, None, :] - split_point_positions_pred[None, :, :], axis=-1)

    # find the minimum cost for each gt split point
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    tp = np.sum(cost_matrix[row_ind, col_ind] < threshold)
    fp = len(split_point_positions_pred) - tp
    fn = len(split_point_positions_gt) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    F1 = f1_score(precision, recall)

    return F1

def render_graph(graph, imsize=[256, 256], width=10, coord_range=None):
    """
    Render a graph as an image.
    Args:
        graph: networkx graph
        imsize: image size [width, height]
        width: line width of edges
        coord_range: [xmin, xmax, ymin, ymax] for coordinate normalization
    Returns: rendered graph
    """
    im = np.zeros(imsize).astype(np.uint8)

    # Auto-detect coordinate range if not provided
    if coord_range is None:
        all_coords = np.array([graph.nodes[n]['pos'] for n in graph.nodes()])
        if len(all_coords) > 0:
            xmin, ymin = all_coords.min(axis=0)
            xmax, ymax = all_coords.max(axis=0)
            # Add padding
            padding = 5
            coord_range = [xmin - padding, xmax + padding, ymin - padding, ymax + padding]
        else:
            coord_range = [0, imsize[0], 0, imsize[1]]
    
    xmin, xmax, ymin, ymax = coord_range
    
    def map_coord(x, y):
        """Map world coordinates to image pixel coordinates"""
        # Normalize to [0, 1]
        x_norm = (x - xmin) / (xmax - xmin + 1e-8)
        y_norm = (y - ymin) / (ymax - ymin + 1e-8)
        # Map to image size
        px = int(x_norm * (imsize[0] - 1))
        py = int(y_norm * (imsize[1] - 1))
        # Clamp to valid range
        px = max(0, min(imsize[0] - 1, px))
        py = max(0, min(imsize[1] - 1, py))
        return px, py

    for e in graph.edges():
        start = graph.nodes[e[0]]['pos']
        end = graph.nodes[e[1]]['pos']
        x1, y1 = map_coord(start[0], start[1])
        x2, y2 = map_coord(end[0], end[1])
        cv2.line(im, (x1, y1), (x2, y2), 255, width)
    return im

def calc_iou(graph_gt, graph_pred, area_size=[256, 256], lane_width=10):
    """
    Calculate IoU of two graphs.
    """
    # Get combined coordinate range for consistent rendering
    all_gt_coords = np.array([graph_gt.nodes[n]['pos'] for n in graph_gt.nodes()]) if len(graph_gt.nodes()) > 0 else np.array([])
    all_pred_coords = np.array([graph_pred.nodes[n]['pos'] for n in graph_pred.nodes()]) if len(graph_pred.nodes()) > 0 else np.array([])
    
    if len(all_gt_coords) > 0 and len(all_pred_coords) > 0:
        all_coords = np.vstack([all_gt_coords, all_pred_coords])
        xmin, ymin = all_coords.min(axis=0)
        xmax, ymax = all_coords.max(axis=0)
        padding = 5
        coord_range = [xmin - padding, xmax + padding, ymin - padding, ymax + padding]
    elif len(all_gt_coords) > 0:
        xmin, ymin = all_gt_coords.min(axis=0)
        xmax, ymax = all_gt_coords.max(axis=0)
        padding = 5
        coord_range = [xmin - padding, xmax + padding, ymin - padding, ymax + padding]
    elif len(all_pred_coords) > 0:
        xmin, ymin = all_pred_coords.min(axis=0)
        xmax, ymax = all_pred_coords.max(axis=0)
        padding = 5
        coord_range = [xmin - padding, xmax + padding, ymin - padding, ymax + padding]
    else:
        coord_range = None
    
    render_gt = render_graph(graph_gt, imsize=area_size, width=lane_width, coord_range=coord_range)
    render_pred = render_graph(graph_pred, imsize=area_size, width=lane_width, coord_range=coord_range)

    # Calculate IoU
    intersection = np.logical_and(render_gt, render_pred)
    union = np.logical_or(render_gt, render_pred)
    iou = np.sum(intersection) / (1e-8 + np.sum(union))

    return iou

# Import APLS from the copied files
try:
    from projects.SeqGrowGraph.seq_grow_graph.apls import execute_apls, prepare_graph
    APLS_AVAILABLE = True
except ImportError:
    APLS_AVAILABLE = False
    
def calc_apls(g_gt, g_pred):
    """
    Calculate APLS (Average Path Length Similarity) metric.
    """
    if not APLS_AVAILABLE:
        warnings.warn("APLS module not available, returning 0.0")
        return 0.0
    
    try:
        # Prepare graphs for APLS calculation
        g_gt_prep = prepare_graph(g_gt.copy())
        g_pred_prep = prepare_graph(g_pred.copy())
        
        # Execute APLS
        apls_dict = execute_apls([g_gt_prep], [g_pred_prep], verbose=False)
        
        return apls_dict['APLS']
    except Exception as e:
        warnings.warn(f"Error calculating APLS: {e}")
        return 0.0 

class GeoTopoEvaluator():
    def __init__(self, gt_graph, pred_graph, interp_dist=2, prop_dist=400, gmode='direct', boundary=None):
        self.gt_graph = gt_graph
        self.prop_graph = pred_graph
        self.interp_dist  = interp_dist
        self.prop_dist = prop_dist
        self.gmode = gmode
        # Set boundary for filtering nodes. Default values from CGNet for NuScenes
        self.boundary = boundary if boundary is not None else {'xmin': -150, 'xmax': 150, 'ymin': -300, 'ymax': 300}

    def interpolateGraph(self, graph):
        newgraph = {}
        exist = set()

        for nid, nei in graph.items():
            for nn in nei:
                if (nid, nn) in exist or (nn, nid) in exist:
                    continue
                x1, y1 = nid
                x2, y2 = nn

                exist.add((nid, nn))

                L = int(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) / self.interp_dist) + 1
                L = max(2, L)
                last_node = (x1, y1)
                for i in range(1, L):
                    a = 1.0 - float(i) / (L - 1)
                    x = x1 * a + x2 * (1 - a)
                    y = y1 * a + y2 * (1 - a)

                    nk1 = last_node
                    nk2 = (x, y)

                    # Boundary checks
                    if x < self.boundary['xmin'] or x >= self.boundary['xmax'] or y < self.boundary['ymin'] or y >= self.boundary['ymax']:
                        last_node = (x, y)
                        continue

                    if last_node[0] < self.boundary['xmin'] or last_node[0] >= self.boundary['xmax'] or last_node[1] < self.boundary['ymin'] or last_node[1] >= self.boundary['ymax']:
                        last_node = (x, y)
                        continue

                    if nk1 not in newgraph:
                        newgraph[nk1] = [nk2]
                    elif nk2 not in newgraph[nk1]:
                        newgraph[nk1].append(nk2)

                    if self.gmode =='direct':
                        if nk2 not in newgraph:
                            newgraph[nk2] = []
                    else:
                        if nk2 not in newgraph:
                            newgraph[nk2] = [nk1]
                        elif nk1 not in newgraph[nk2]:
                            newgraph[nk2].append(nk1)

                    last_node = (x, y)

        return newgraph

    def propagateByDistance(self, graph, nid, steps=400):
        visited = set()
        queue = [(nid, 0)]

        def distance(p1, p2):
            a = (p1[0] - p2[0]) ** 2
            b = (p1[1] - p2[1]) ** 2
            return np.sqrt(a + b)

        while len(queue) > 0:
            cur_nid, depth = queue.pop()
            visited.add(cur_nid)

            if depth >= steps:
                continue

            if cur_nid in graph:
                for nei in graph[cur_nid]:
                    if nei in visited:
                        continue
                    queue.append((nei, depth + distance(cur_nid, nei)))

        return list(visited)

    def match(self, nodes1, nodes2, thr=8):
        idx = rtree.index.Index()
        for i in range(len(nodes1)):
            x, y = nodes1[i]
            idx.insert(i, (x - 1, y - 1, x + 1, y + 1))

        pairs = []
        m = thr

        for i in range(len(nodes2)):
            x, y = nodes2[i]
            candidates = list(idx.intersection((x - m, y - m, x + m, y + m)))
            for n in candidates:
                x2, y2 = nodes1[n]
                r = (x2 - x) ** 2 + (y2 - y) ** 2
                if r < thr * thr:
                    pairs.append((i, n, r))

        pairs = sorted(pairs, key=lambda x: x[2])

        matched = 0
        prop_set = set()
        gt_set = set()

        for pair in pairs:
            n1, n2, _ = pair
            if n1 in prop_set or n2 in gt_set:
                continue

            prop_set.add(n1)
            gt_set.add(n2)
            matched += 1

        precision = float(matched) / len(nodes1) if len(nodes1) > 0 else 0
        recall = float(matched) / len(nodes2) if len(nodes2) > 0 else 0

        return precision, recall

    def topoMetric(self, thr=8, mask=None, verbose=False):
        prop_graph = self.interpolateGraph(self.prop_graph)
        gt_graph = self.interpolateGraph(self.gt_graph)

        prop_nodes = list(prop_graph.keys())
        gt_nodes = list(gt_graph.keys())

        if len(prop_nodes)==0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        idx = rtree.index.Index()

        for i in range(len(gt_nodes)):
            x, y = gt_nodes[i]
            idx.insert(i, (x - 1, y - 1, x + 1, y + 1))
        
        pairs = []
        m = thr

        for i in range(len(prop_nodes)):
            x, y = prop_nodes[i]
            candidates = list(idx.intersection((x - m, y - m, x + m, y + m)))
            for n in candidates:
                x2, y2 = gt_nodes[n]
                r = (x2 - x) ** 2 + (y2 - y) ** 2
                if r < thr * thr:
                    pairs.append((i, n, r))

        pairs = sorted(pairs, key=lambda x: x[2])

        matched = 0
        prop_set = set()
        gt_set = set()
        ps, rs = [], []
        jps, jrs = [], []
        
        for pair in pairs:
            n1, n2, _ = pair
            if n1 in prop_set or n2 in gt_set:
                continue

            prop_set.add(n1)
            gt_set.add(n2)

            # compute precision and recall
            if gt_nodes[n2] in gt_graph and len(gt_graph[gt_nodes[n2]]) > 1:
                nodes1 = self.propagateByDistance(prop_graph, prop_nodes[n1], self.prop_dist)
                nodes2 = self.propagateByDistance(gt_graph, gt_nodes[n2], self.prop_dist)

                p, r = self.match(nodes1, nodes2, thr=thr)
                jps.append(p)
                jrs.append(r)

            if matched % 10 == 0:
                nodes1 = self.propagateByDistance(prop_graph, prop_nodes[n1], self.prop_dist)
                nodes2 = self.propagateByDistance(gt_graph, gt_nodes[n2], self.prop_dist)

                p, r = self.match(nodes1, nodes2, thr=thr)
                ps.append(p)
                rs.append(r)

            matched += 1

        geo_precision = float(matched) / len(prop_nodes) if len(prop_nodes) > 0 else 0
        geo_recall = float(matched) / len(gt_nodes) if len(gt_nodes) > 0 else 0
        geo_f1 = f1_score(geo_precision, geo_recall)

        topo_precision = float(matched) / len(prop_nodes) * np.mean(ps) if len(prop_nodes) > 0 and len(ps) > 0 else 0
        topo_recall = float(matched) / len(gt_nodes) * np.mean(rs) if len(gt_nodes) > 0 and len(rs) > 0 else 0
        topo_f1 = f1_score(topo_precision, topo_recall)

        jtopo_precision = float(matched) / len(prop_nodes) * np.mean(jps) if len(prop_nodes) > 0 and len(jps) > 0 else 0
        jtopo_recall = float(matched) / len(gt_nodes) * np.mean(jrs) if len(gt_nodes) > 0 and len(jrs) > 0 else 0
        jtopo_f1 = f1_score(jtopo_precision, jtopo_recall)

        return geo_precision, geo_recall, topo_precision, topo_recall, geo_f1, topo_f1, jtopo_f1

# --- Main Class ---

class LaneDiffusionMetric:
    def __init__(self, area_size=[256, 256], lane_width=10, interp_dist=2, prop_dist=400, boundary=None):
        self.area_size = area_size
        self.lane_width = lane_width
        self.interp_dist = interp_dist
        self.prop_dist = prop_dist
        self.boundary = boundary

    def convert_to_networkx(self, eval_seq2graph):
        """
        Convert EvalSeq2Graph object to networkx.DiGraph.
        """
        G = nx.DiGraph()
        
        # Add nodes
        for node in eval_seq2graph.graph_nodelist:
            # Use node index as ID, store position as tuple of floats
            # Ensure coord is converted to proper format (handle numpy arrays)
            if isinstance(node.coord, np.ndarray):
                pos = (float(node.coord[0]), float(node.coord[1]))
            else:
                pos = (float(node.coord[0]), float(node.coord[1]))
            G.add_node(node.index, pos=pos)
            
        # Add edges
        for node in eval_seq2graph.graph_nodelist:
            for child_node, _ in node.childs:
                # Add edge from node to child
                G.add_edge(node.index, child_node.index)
                
        return G

    def nx_to_geo_topo_format(self, nx_graph):
        """
        Convert a networkx graph to the format used for calculating the GEO and TOPO metrics.
        Returns a dict: {(x, y): [(x_nei, y_nei), ...]}
        """
        neighbors = {}

        for e in nx_graph.edges():
            # Check if nodes exist (they should)
            if e[0] not in nx_graph.nodes or e[1] not in nx_graph.nodes:
                continue
                
            p1 = nx_graph.nodes[e[0]]['pos']
            p2 = nx_graph.nodes[e[1]]['pos']
            
            # Use tuple (x, y) as key - multiply by 10 and convert to int to match CGNet format
            k1 = (int(p1[0]*10), int(p1[1]*10))
            k2 = (int(p2[0]*10), int(p2[1]*10))

            if k1 not in neighbors:
                neighbors[k1] = []

            if k2 not in neighbors[k1]:
                neighbors[k1].append(k2)
        
        return neighbors

    def evaluate(self, gt_seq2graph, pred_seq2graph):
        """
        Calculate all metrics for a single sample.
        """
        # Convert to networkx
        g_gt = self.convert_to_networkx(gt_seq2graph)
        g_pred = self.convert_to_networkx(pred_seq2graph)

        # 1. IoU
        iou = calc_iou(g_gt, g_pred, area_size=self.area_size, lane_width=self.lane_width)

        # 2. SDA
        sda = calc_sda(g_gt, g_pred)

        # 3. APLS (Placeholder)
        apls = calc_apls(g_gt, g_pred)

        # 4. GEO & TOPO F1
        # Convert to dictionary format for GeoTopoEvaluator
        gt_dict = self.nx_to_geo_topo_format(g_gt)
        pred_dict = self.nx_to_geo_topo_format(g_pred)

        evaluator = GeoTopoEvaluator(gt_dict, pred_dict, interp_dist=self.interp_dist, prop_dist=self.prop_dist, boundary=self.boundary)
        geo_p, geo_r, topo_p, topo_r, geo_f1, topo_f1, jtopo_f1 = evaluator.topoMetric()

        return {
            'iou': iou,
            'sda': sda,
            'apls': apls,
            'geo_f1': geo_f1,
            'topo_f1': topo_f1,
            'jtopo_f1': jtopo_f1
        }
