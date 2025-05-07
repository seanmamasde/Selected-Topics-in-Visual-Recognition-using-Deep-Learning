"""
Train ViTDet Mask R-CNN on the dataset, identical flow to train.py except model.
"""

import argparse
import os

from detectron2 import model_zoo
from detectron2.config import LazyConfig, get_cfg
from detectron2.engine import DefaultTrainer, default_setup, launch
from detectron2.evaluation import COCOEvaluator

import src.utils.soft_nms
from src.config import NUM_CLASSES, OUTPUT_DIR
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


def setup(args):
    cfg = get_cfg()
    # lcfg = LazyConfig.load(args.config_file)
    # cfg.__dict__.update(lcfg)

    cfg.MODEL.WEIGHTS = (
        "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
        "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    )
    # cfg.MODEL.WEIGHTS = lcfg.train.init_checkpoint
    # cfg.MODEL.BACKBONE.NAME = lcfg.model.backbone._target_
    cfg.MODEL.DEVICE = "cuda"
    cfg.INPUT.MASK_FORMAT = "bitmask"

    cfg.DATASETS.TRAIN = ("cells_train",)
    cfg.DATASETS.TEST = ("cells_val",)
    cfg.DATALOADER.NUM_WORKERS = 16

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES

    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 1e-4
    cfg.SOLVER.MAX_ITER = 1_000_000
    cfg.SOLVER.STEPS = (7000, 9000)
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000

    cfg.INPUT.MIN_SIZE_TRAIN = (400, 600, 800, 1000)
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 0
    cfg.INPUT.MAX_SIZE_TEST = 1024

    cfg.TEST.EVAL_PERIOD = 1000
    cfg.TEST.AUG.ENABLED = True
    cfg.TEST.AUG.FLIP = True
    cfg.TEST.AUG.MIN_SIZES = (400, 600, 800)
    cfg.TEST.AUG.MAX_SIZE = 1333

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--dist-url", default="auto")
    args = parser.parse_args()

    launch(
        main,
        args.num_gpus,
        num_machines=1,
        machine_rank=0,
        dist_url=args.dist_url,
        args=(args,),
    )
