import torch
from detectron2.structures import (
    Boxes,
    RotatedBoxes,
    pairwise_iou,
    pairwise_iou_rotated,
)

"""
Soft-NMS implementation for Detectron2.
Based on: https://github.com/facebookresearch/detectron2/pull/1183/files
Courtesy of the regionclip-demo HuggingFace repo :contentReference[oaicite:0]{index=0}.
"""


def soft_nms(
    boxes,
    scores,
    method="linear",
    gaussian_sigma=0.5,
    linear_threshold=0.3,
    prune_threshold=0.001,
):
    """
    Soft Non-Maximum Suppression for axis-aligned boxes.
    Args:
      boxes (Tensor[N, 4]): in (x1, y1, x2, y2) format
      scores (Tensor[N]): confidence scores
      method (str): 'linear', 'gaussian', or 'hard'
      gaussian_sigma (float): parameter for Gaussian penalty
      linear_threshold (float): IoU threshold for linear decay
      prune_threshold (float): score threshold for pruning
    Returns:
      kept_indices (Tensor): indices kept after soft-NMS
      kept_scores (Tensor): re-scored scores of kept boxes
    """
    return _soft_nms(
        Boxes,
        pairwise_iou,
        boxes,
        scores,
        method,
        gaussian_sigma,
        linear_threshold,
        prune_threshold,
    )


def soft_nms_rotated(
    boxes,
    scores,
    method="linear",
    gaussian_sigma=0.5,
    linear_threshold=0.3,
    prune_threshold=0.001,
):
    """
    Soft-NMS for rotated boxes.
    Boxes format: (x_ctr, y_ctr, width, height, angle_degrees)
    """
    return _soft_nms(
        RotatedBoxes,
        pairwise_iou_rotated,
        boxes,
        scores,
        method,
        gaussian_sigma,
        linear_threshold,
        prune_threshold,
    )


def batched_soft_nms(
    boxes,
    scores,
    idxs,
    method="linear",
    gaussian_sigma=0.5,
    linear_threshold=0.3,
    prune_threshold=0.001,
):
    """
    Batched Soft-NMS: applies per-class by offsetting boxes.
    Args:
      boxes (Tensor[N, 4]), scores (Tensor[N]), idxs (Tensor[N]): class indices
    """
    if boxes.numel() == 0:
        return (
            torch.empty((0,), dtype=torch.int64, device=boxes.device),
            torch.empty((0,), dtype=torch.float32, device=scores.device),
        )
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    return soft_nms(
        boxes_for_nms,
        scores,
        method,
        gaussian_sigma,
        linear_threshold,
        prune_threshold,
    )


def batched_soft_nms_rotated(
    boxes,
    scores,
    idxs,
    method="linear",
    gaussian_sigma=0.5,
    linear_threshold=0.3,
    prune_threshold=0.001,
):
    """
    Batched Soft-NMS for rotated boxes.
    """
    if boxes.numel() == 0:
        return (
            torch.empty((0,), dtype=torch.int64, device=boxes.device),
            torch.empty((0,), dtype=torch.float32, device=scores.device),
        )
    # offset centers sufficiently by class
    max_coord = boxes[:, :2].max() + torch.norm(boxes[:, 2:4], 2, dim=1).max()
    offsets = idxs.to(boxes) * (max_coord + 1)
    boxes_for_nms = boxes.clone()
    boxes_for_nms[:, :2] += offsets[:, None]
    return soft_nms_rotated(
        boxes_for_nms,
        scores,
        method,
        gaussian_sigma,
        linear_threshold,
        prune_threshold,
    )


def _soft_nms(
    box_class,
    pairwise_iou_func,
    boxes,
    scores,
    method,
    gaussian_sigma,
    linear_threshold,
    prune_threshold,
):
    """
    Core Soft-NMS loop.
    """
    boxes = boxes.clone()
    scores = scores.clone()
    idxs = torch.arange(scores.size(0), device=boxes.device)
    keep_indices = []
    keep_scores = []

    while scores.numel() > 0:
        top_idx = torch.argmax(scores)
        keep_indices.append(idxs[top_idx].item())
        keep_scores.append(scores[top_idx].item())

        top_box = boxes[top_idx].unsqueeze(0)
        ious = pairwise_iou_func(box_class(top_box), box_class(boxes))[0]

        if method == "linear":
            decay = torch.ones_like(ious)
            mask = ious > linear_threshold
            decay[mask] = 1 - ious[mask]
        elif method == "gaussian":
            decay = torch.exp(-(ious**2) / gaussian_sigma)
        elif method == "hard":
            decay = (ious < linear_threshold).float()
        else:
            raise NotImplementedError(
                f"{method} is not a valid Soft-NMS method"
            )

        scores = scores * decay
        keep_mask = scores > prune_threshold
        keep_mask[top_idx] = False
        boxes = boxes[keep_mask]
        scores = scores[keep_mask]
        idxs = idxs[keep_mask]

    return (
        torch.tensor(keep_indices, dtype=torch.int64, device=boxes.device),
        torch.tensor(keep_scores, dtype=torch.float32, device=scores.device),
    )
