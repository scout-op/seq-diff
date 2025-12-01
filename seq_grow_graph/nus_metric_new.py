import sys
import argparse
from projects.SeqGrowGraph.seq_grow_graph.nus_reach_metric import NuScenesReachMetric
from mmengine import Config, load
from mmengine.logging import MMLogger
from os import path as osp
from typing import Dict, List, Optional, Sequence, Tuple, Union
import json
import numpy as np
from tqdm import tqdm
from projects.SeqGrowGraph.seq_grow_graph.lane_diffusion_metric import LaneDiffusionMetric
from projects.SeqGrowGraph.seq_grow_graph.transforms.loading import LoadNusOrderedBzCenterline, TransformGraph2Seq
from projects.SeqGrowGraph.seq_grow_graph.core.centerline.structures.ljc_bz_centerline import EvalSeq2Graph_with_start
from projects.SeqGrowGraph.seq_grow_graph.transforms.roadnet_reach_dist_eval_new import get_geom, get_range

data_root = "./data/nuscenes/"

# grid_conf = dict(
#         xbound=[1.0, 50.0, 0.5],
#         ybound=[-25.0, 25.0, 0.5],
#         zbound=[-10.0, 10.0, 20.0],
#         dbound=[4.0, 48.0, 1.0],)

grid_conf = dict(
    xbound=[-48.0, 48.0, 0.5],
    ybound=[-32.0, 32.0, 0.5],
    zbound=[-10.0, 10.0, 20.0],
    dbound=[4.0, 48.0, 1.0],
)

bz_grid_conf = dict(
    xbound=[-55.0, 55.0, 0.5],
    ybound=[-55.0, 55.0, 0.5],
    zbound=[-10.0, 10.0, 20.0],
    dbound=[4.0, 48.0, 1.0],
)

backend_args = None

params = dict(
    data_root=data_root,
    ann_file=data_root + "nuscenes_centerline_infos_val.pkl",
    metric="ar_reach",
    backend_args=backend_args,
    grid_conf=grid_conf,
    bz_grid_conf=bz_grid_conf,
    jsonfile_prefix="save_result",
    landmark_thresholds=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    reach_thresholds=[1, 2, 3, 4, 5],
    is_new=True,
)

class NewNuScenesReachMetric(NuScenesReachMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.grid_conf = kwargs.get('grid_conf')
        self.bz_grid_conf = kwargs.get('bz_grid_conf')

    def evaluate_lane_diffusion(self, result_path, logger=None):
        if logger:
            logger.info("Evaluating LaneDiffusion metrics...")
        else:
            print("Evaluating LaneDiffusion metrics...")
        
        # Initialize metric calculator
        ld_metric = LaneDiffusionMetric(
            area_size=[256, 256], 
            lane_width=10,
            interp_dist=2,
            prop_dist=400
        )

        # Load results
        with open(result_path) as f:
            results = json.load(f)["results"]

        # Load GT loader
        centerline_loader = LoadNusOrderedBzCenterline(self.grid_conf, self.bz_grid_conf)
        centerline_transform = TransformGraph2Seq()
        
        # Geometry info
        dx, bx, nx, pc_range, ego_points = get_geom(self.grid_conf)
        bz_dx, bz_bx, bz_nx, bz_pc_range = get_range(self.bz_grid_conf)

        metrics_sum = {
            'iou': 0.0, 'sda': 0.0, 'apls': 0.0,
            'geo_f1': 0.0, 'topo_f1': 0.0, 'jtopo_f1': 0.0
        }
        count = 0

        for info in tqdm(self.data_infos):
            token = info['token']
            if token not in results:
                continue
                
            try:
                # Load GT
                info = centerline_loader(info)
                info = centerline_transform(info)
                gt_sequence = info["centerline_sequence"]
                
                gt_nodegraph = EvalSeq2Graph_with_start(
                    token, gt_sequence, pc_range, dx, bz_pc_range, bz_dx
                )
                
                # Load Pred
                pred_sequence = results[token]
                pred_nodegraph = EvalSeq2Graph_with_start(
                    token, pred_sequence, pc_range, dx, bz_pc_range, bz_dx
                )
                
                # Calculate metrics
                sample_metrics = ld_metric.evaluate(gt_nodegraph, pred_nodegraph)
                
                for k, v in sample_metrics.items():
                    if not np.isnan(v):
                        metrics_sum[k] += v
                count += 1
            except Exception as e:
                if logger:
                    logger.warning(f"Error evaluating sample {token}: {e}")
                else:
                    print(f"Error evaluating sample {token}: {e}")
                continue

        # Average
        metrics_avg = {k: v / count if count > 0 else 0.0 for k, v in metrics_sum.items()}
        
        for k, v in metrics_avg.items():
            if logger:
                logger.info(f"LaneDiffusion {k}: {v:.4f}")
            else:
                print(f"LaneDiffusion {k}: {v:.4f}")
            
        return metrics_avg

    def compute_metrics(self, result_path) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (List[dict]): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        print(result_path)
        # classes = self.dataset_meta['classes']
        # self.version = self.dataset_meta['version']
        # load annotations
        self.data_infos = load(self.ann_file, backend_args=self.backend_args)["infos"]
        # result_path, tmp_dir = self.format_results(results, classes,
        #    self.jsonfile_prefix)

        metric_dict = {}

        if self.format_only:
            logger.info(f"results are saved in {osp.basename(self.jsonfile_prefix)}")
            return metric_dict

        for metric in self.metrics:
            ap_dict = self._evaluate_single(result_path, metric=metric, logger=logger)
            for result in ap_dict:
                metric_dict[result] = ap_dict[result]

        # Evaluate LaneDiffusion metrics
        ld_metrics = self.evaluate_lane_diffusion(result_path, logger=logger)
        metric_dict.update(ld_metrics)

        # if tmp_dir is not None:
        #     tmp_dir.cleanup()
        return metric_dict

def main():
    parser = argparse.ArgumentParser(description='Compute NuScenes Reach Metrics')
    parser.add_argument('--result_path', type=str, required=True,
                        help='Path to the results JSON file')
    
    args = parser.parse_args()
    
    metric = NewNuScenesReachMetric(**params)
    metric.compute_metrics(result_path=args.result_path)

if __name__ == "__main__":
    main()