import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_model(
    num_classes: int = 11,
    *,
    device: str | torch.device | None = None,
    pretrained: bool = True,
) -> torch.nn.Module:
    """
    Build a Faster-RCNN-ResNet50-FPN detector.

    Parameters
    ----------
    num_classes : int
        Number of classes *including* background.
    device : str | torch.device | None
        Move the model to this device.  Default: CUDA if available else CPU.
    pretrained : bool
        Load COCO-pretrained weights.
    """
    device = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = fasterrcnn_resnet50_fpn_v2(
        weights="DEFAULT" if pretrained else None
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model.to(device)
