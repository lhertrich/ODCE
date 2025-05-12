import torch
import torchmetrics

from detr.models.detr import SetCriterion
from detr.models.matcher import HungarianMatcher

from PIL import Image, ImageDraw, ImageFont


def inference(
    image: Image,
    image_feature: torch.Tensor,
    ground_truth_box,
    model,
    device,
    threshold: float = 0.5,
    output_path: str = "output.png",
):
    """
    Return the class label and bounding box for a single image.
    Ground truth box (from Ref-L4 dataset): xywh format.
    """
    model.eval()
    model.to(device)
    image_feature = image_feature.to(device)

    with torch.no_grad():
        raw_logits, raw_pred_boxes = model.forward(
            image_feature.view(1, -1).unsqueeze(1)
        )

    raw_pred_boxes = raw_pred_boxes.sigmoid()
    probs = raw_logits.softmax(-1)[0]  # shape: [num_queries, num_classes]
    scores, labels = probs.max(-1)  # shape: [num_queries]

    keep = scores > threshold
    boxes = raw_pred_boxes[0][keep]  # shape: [num_kept_queries, 4]
    labels = labels[keep]
    scores = scores[keep]

    # Convert predicted boxes from [cx, cy, w, h] to [x0, y0, x1, y1]
    w, h = image.size
    boxes_xyxy = []

    for box in boxes:
        cx, cy, bw, bh = box
        x0 = (cx - bw / 2) * w
        y0 = (cy - bh / 2) * h
        x1 = (cx + bw / 2) * w
        y1 = (cy + bh / 2) * h
        boxes_xyxy.append([x0.item(), y0.item(), x1.item(), y1.item()])

    # Convert ground truth box from xywh to xyxy
    x, y, w, h = ground_truth_box
    ground_truth_box = [x, y, x + w, y + h]

    draw = ImageDraw.Draw(image)

    for box, label_id, score in zip(boxes_xyxy, labels, scores):
        x0, y0, x1, y1 = box
        color = "red"
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        draw.text((x0, y0), f"{label_id}: {score:.2f}", fill=color)

    draw.rectangle(ground_truth_box, outline="green", width=3)

    image.save(output_path)  # Save the image with bounding boxes

    return image


##### HELPERS
def get_loss_fn(config):
    # 1. Initialize the matcher
    matcher = HungarianMatcher(
    cost_class=1.0,
    cost_bbox=5.0,
    cost_giou=2.0
    )

    weight_dict = {
        'loss_ce':   1.0,   # classification
        'loss_bbox': 5.0,   # L1
        'loss_giou': 2.0,   # GIoU
    }
    eos_coef = 0.1  # weight for no-object class
    criterion = SetCriterion(
        num_classes=config.num_classes, #+1,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=eos_coef,
        losses=['labels','boxes']
    )
    return criterion, weight_dict

def convert_boxes(boxes, image_size=None):
    if boxes.dim() == 1:
        boxes = boxes.unsqueeze(0)
    
    if image_size is not None:
        width, height = image_size
        boxes = boxes.clone()
        boxes[:, 0] /= width
        boxes[:, 1] /= height
        boxes[:, 2] /= width
        boxes[:, 3] /= height

    # Convert top-left to center coordinates
    cx = boxes[:, 0] + boxes[:, 2] / 2
    cy = boxes[:, 1] + boxes[:, 3] / 2

    return torch.stack([cx, cy, boxes[:, 2], boxes[:, 3]], dim=1)


def eval_model(
        config,
        model,
        dataset,
        device,
):
    
    criterion, weight_dict = get_loss_fn(config)
    criterion = criterion.to(device)

    map_calculator = torchmetrics.detection.MeanAveragePrecision(box_format='xyxy')
    map_calculator = map_calculator.to(device)

    successful_hits = 0
    total_images_with_gt_objects = 0

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.data.batch_size, shuffle=False)
    device = torch.device(config.device)
    model.eval()
    model.to(device)
    val_loss = 0.0
    count = 0
    with torch.no_grad():
        for images, data in dataloader:

            inputs = data['clip_vision'].to(device)
            raw_logits, raw_pred_boxes = model(inputs.unsqueeze(1))
                
            pred_boxes = raw_pred_boxes.sigmoid()
            labels = data["labels"].to(device)
            boxes   = data["bbox"].to(device)            # shape (B,4)
            widths  = data["width"].tolist()             # [w1, w2, …]
            heights = data["height"].tolist()            # [h1, h2, …]

            # 2) normalize & convert each sample’s box to (cx,cy,w,h) in [0,1]
            converted_boxes = [
                convert_boxes(box, (w, h)) 
                for box, w, h in zip(boxes.unbind(0), widths, heights)
            ]

            processed_labels = []
            for lbl in labels.unbind(0):
                # If label is scalar (0-dim), convert to 1D tensor
                if lbl.dim() == 0:
                    lbl = lbl.unsqueeze(0)
                # If label is empty, add a default "no object" label (assuming class 0 is background)
                #TODO check if 0 is background class
                if lbl.numel() == 0:
                    lbl = torch.tensor([0], device=lbl.device)
                processed_labels.append(lbl)

            targets = [
                {"labels": lbl,    # (n_obj,)
                "boxes":  box}   # (n_obj,4), now in cxcywh normalized coords
                for lbl, box in zip(processed_labels, converted_boxes)
            ]

            outputs = {
                "pred_logits": raw_logits,
                "pred_boxes":  pred_boxes
            }
            loss_dict = criterion(outputs, targets)
            weighted_loss_dict = {k: loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict}
            total_loss = sum(weighted_loss_dict.values())
            val_loss += total_loss.item()


            for i in range(len(raw_logits)):
                


            count    += 1

    