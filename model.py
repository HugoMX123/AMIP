from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch import save
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
import time

def get_model(num_classes):
    start_time = time.time()
    print("Loading Mask R-CNN model...")
    
    # Use Mask R-CNN instead of Faster R-CNN
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)
    print(f"Model backbone loaded in {time.time() - start_time:.2f} seconds")
    
    # Save the initial model state
    save(model.state_dict(), "/net/travail/hsosapavonca/AMIP/model/default/prebuilt_model.pth")
    
    # Modify the box predictor
    start_time = time.time()
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)
    print(f"Box predictor modified in {time.time() - start_time:.2f} seconds")
    
    # Modify the mask predictor
    start_time = time.time()
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    print(f"Mask predictor modified in {time.time() - start_time:.2f} seconds")
    
    return model

