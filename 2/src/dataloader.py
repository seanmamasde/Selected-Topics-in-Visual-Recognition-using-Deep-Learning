import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


class CocoDetectionDataset(Dataset):
    """
    Custom dataset for COCO detection format.
    Args:
        image_dir (str): Directory with all the images.
        annotation_file (str): Path to the COCO format annotation file.
        transforms (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, image_dir, annotation_file, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms

        # Load COCO format annotations
        with open(annotation_file, "r") as f:
            coco = json.load(f)

        # Index for lookup
        self.image_id_to_info = {img["id"]: img for img in coco["images"]}
        self.annotations = coco["annotations"]
        self.categories = {
            cat["id"]: cat["name"] for cat in coco["categories"]
        }

        # Group annotations by image_id
        from collections import defaultdict

        self.image_to_anns = defaultdict(list)
        for ann in self.annotations:
            self.image_to_anns[ann["image_id"]].append(ann)

        self.image_ids = list(self.image_id_to_info.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.image_id_to_info[image_id]
        img_path = os.path.join(self.image_dir, img_info["file_name"])

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Get annotations for this image
        anns = self.image_to_anns[image_id]

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])
            areas.append(ann["area"])
            iscrowd.append(ann.get("iscrowd", 0))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.uint8)
        image_id = torch.tensor([image_id])

        target = {
            "boxes": boxes,
            "labels": labels,
            "area": areas,
            "iscrowd": iscrowd,
            "image_id": image_id,
        }

        if self.transforms:
            image, target = self.transforms(image, target)
        else:
            image = F.to_tensor(image)

        return image, target
