"""
Train Cascade Mask RCNN on the dataset.
"""

import argparse
import os

from detectron2 import model_zoo
from detectron2.config import LazyConfig, get_cfg, instantiate
from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.engine import DefaultTrainer, default_setup, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.projects import point_rend

import src.utils.soft_nms
from src.config import CASCADE_YAML, NUM_CLASSES, OUTPUT_DIR
from src.dataset import register_cell_dataset


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

    @classmethod
    def build_model(cls, cfg):
        model = super().build_model(cfg)
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Trainer] ✨ Trainable parameters: {total:,}")
        return model

    @classmethod
    def build_train_loader(cls, cfg):
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            ),
        ]
        mapper = DatasetMapper(cfg, is_train=True, augmentations=augs)
        return build_detection_train_loader(cfg, mapper=mapper)


def setup(args):
    cfg = get_cfg()
    if args.model == "grcnn":
        cfg.merge_from_file(model_zoo.get_config_file(CASCADE_YAML))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(CASCADE_YAML)
        # cfg.MODEL.RESNETS.DEPTH = 101
        # cfg.MODEL.FPN.OUT_CHANNELS = 256
    elif args.model == "pointrend":
        point_rend.add_pointrend_config(cfg)
        cfg.merge_from_file(
            # "/workspace/detectron2/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml"
            "/content/detectron2/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml"
        )
        cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco/28119989/model_final_ba17b9.pkl"

    cfg.MODEL.DEVICE = "cuda"
    cfg.INPUT.MASK_FORMAT = "bitmask"

    cfg.DATASETS.TRAIN = ("cells_train",)
    cfg.DATASETS.TEST = ("cells_val",)
    cfg.DATALOADER.NUM_WORKERS = 16

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.MODEL.ROI_HEADS.CASCADE_BBOX_REG_WEIGHTS = (
        (1.0, 1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0, 1.0),
    )
    cfg.MODEL.ROI_HEADS.CASCADE_BBOX_REG_LOSSES = (
        "smooth_l1",
        "smooth_l1",
        "smooth_l1",
    )
    cfg.MODEL.ROI_HEADS.CASCADE_IOUS = (0.5, 0.6, 0.7)

    if args.model == "pointrend":
        cfg.MODEL.POINT_HEAD.NUM_CLASSES = NUM_CLASSES

    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 1e-4
    cfg.SOLVER.MAX_ITER = 20000
    cfg.SOLVER.STEPS = (7000, 9000)
    cfg.SOLVER.CHECKPOINT_PERIOD = 2000

    # cfg.INPUT.MIN_SIZE_TRAIN = (512,)
    # cfg.INPUT.MAX_SIZE_TRAIN = 1024
    cfg.INPUT.MIN_SIZE_TRAIN = (
        400,
        600,
        800,
        1000,
    )  # random choice @ each iter
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 0
    cfg.INPUT.MAX_SIZE_TEST = 1024

    cfg.TEST.EVAL_PERIOD = 1000
    # test time augmentation
    cfg.TEST.AUG.ENABLED = True
    cfg.TEST.AUG.FLIP = True
    cfg.TEST.AUG.MIN_SIZES = (400, 500, 600, 700, 800, 900)
    cfg.TEST.AUG.MAX_SIZE = 1600

    cfg.OUTPUT_DIR = str(OUTPUT_DIR.resolve())
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    register_cell_dataset()
    cfg = setup(args)
    trainer = Trainer(cfg)
    if args.eval_only:
        trainer.resume_or_load(resume=args.resume)
        trainer.test(cfg, trainer.model)
        return
    trainer.resume_or_load(resume=args.resume)
    trainer.train()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--dist-url", default="auto")
    args = p.parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=1,
        machine_rank=0,
        dist_url=args.dist_url,
        args=(args,),
    )
