import unittest
import numpy as np
import networkx as nx
from projects.SeqGrowGraph.seq_grow_graph.lane_diffusion_metric import LaneDiffusionMetric

class MockNode:
    def __init__(self, index, coord):
        self.index = index
        self.coord = np.array(coord)
        self.childs = []

class MockEvalSeq2Graph:
    def __init__(self, nodes, edges):
        self.graph_nodelist = []
        node_map = {}
        for idx, coord in nodes.items():
            node = MockNode(idx, coord)
            self.graph_nodelist.append(node)
            node_map[idx] = node
        
        for u, v in edges:
            if u in node_map and v in node_map:
                # childs is list of (child_node, coeff)
                node_map[u].childs.append((node_map[v], None))

class TestLaneDiffusionMetric(unittest.TestCase):
    def setUp(self):
        self.metric = LaneDiffusionMetric()

    def test_convert_to_networkx(self):
        nodes = {1: [0, 0], 2: [10, 10]}
        edges = [(1, 2)]
        mock_graph = MockEvalSeq2Graph(nodes, edges)
        
        nx_graph = self.metric.convert_to_networkx(mock_graph)
        
        self.assertEqual(len(nx_graph.nodes), 2)
        self.assertEqual(len(nx_graph.edges), 1)
        self.assertTrue(nx_graph.has_edge(1, 2))
        np.testing.assert_array_equal(nx_graph.nodes[1]['pos'], np.array([0, 0]))

    def test_perfect_match(self):
        nodes = {1: [0, 0], 2: [10, 10], 3: [20, 20]}
        edges = [(1, 2), (2, 3)]
        gt_graph = MockEvalSeq2Graph(nodes, edges)
        pred_graph = MockEvalSeq2Graph(nodes, edges)
        
        metrics = self.metric.evaluate(gt_graph, pred_graph)
        
        self.assertAlmostEqual(metrics['iou'], 1.0, places=4)
        self.assertAlmostEqual(metrics['geo_f1'], 1.0, places=4)
        self.assertAlmostEqual(metrics['topo_f1'], 1.0, places=4)

    def test_empty_graph(self):
        nodes = {}
        edges = []
        gt_graph = MockEvalSeq2Graph(nodes, edges)
        pred_graph = MockEvalSeq2Graph(nodes, edges)
        
        metrics = self.metric.evaluate(gt_graph, pred_graph)
        
        self.assertEqual(metrics['geo_f1'], 0.0)

if __name__ == '__main__':
    unittest.main()
