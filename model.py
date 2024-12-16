from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.nn as nn
from torch import save
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
import time

def get_model(num_classes):
    start_time = time.time()
    print("Loading Mask R-CNN model...")
    
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    print(f"Model backbone loaded in {time.time() - start_time:.2f} seconds")
    save(model.state_dict(), "/net/travail/hsosapavonca/AMIP/model/default/prebuilt_model.pth")
    
    start_time = time.time()
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)
    print(f"Box predictor modified in {time.time() - start_time:.2f} seconds")
    
    start_time = time.time()
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = nn.Sequential(
        nn.Conv2d(in_features_mask, hidden_layer, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(hidden_layer, num_classes, kernel_size=1)
    )
    print(f"Mask predictor modified in {time.time() - start_time:.2f} seconds")
    
    return model
