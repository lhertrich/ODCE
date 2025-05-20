import sys
import os
#Set parent path to the root of the repo, to be able to import models
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, parent_dir)

import torch
import torchvision
import torchmetrics
import torchvision.ops

# Add the 'detr' package's root to sys.path for its inner imports to work
detr_root = os.path.join(parent_dir, 'detr') # ~/ODCE/detr/
sys.path.insert(0, detr_root)

from detr.models.detr import SetCriterion
from detr.models.matcher import HungarianMatcher

import tarfile
from PIL import Image
import io
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from collections import Counter
from tqdm import tqdm


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

def cxcywh_to_xyxy(boxes_cxcywh):
    cx, cy, w, h = boxes_cxcywh.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack((x1, y1, x2, y2), dim=-1)


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
        iou_threshold,
        with_images = True
):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.data.batch_size, shuffle=False)
    device = torch.device(config.device)
    
    criterion, weight_dict = get_loss_fn(config)
    criterion = criterion.to(device)

    case1_counts = 0  # GT class matches predicted class, bounding boxes overlap
    case2_counts = 0  # Overlapping bounding box, right class, but not the same prediction
    case3_counts = 0  # Overlapping bounding box, but not the right class
    case4_counts = 0  # Right class in prediction, but no overlapping box
    total_gt_objects_in_dataset = 0 
    
    model.eval()
    model.to(device)
    val_loss = 0.0
    count = 0
    with torch.no_grad():
        for images, data in tqdm(dataloader):

            if with_images:
                inputs = images.to(device)
            else:
                inputs = data['clip_vision'].to(device)
            raw_logits, raw_pred_boxes = model(inputs)
                
            pred_boxes = raw_pred_boxes
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
                processed_labels.append(lbl) # Append the label (potentially empty tensor)

            targets = []
            for i in range(len(processed_labels)):
                current_labels = processed_labels[i]
                current_boxes = converted_boxes[i] # Assumes converted_boxes has corresponding shape
                # Ensure box tensor is also empty if labels are empty
                if current_labels.numel() == 0:
                     if current_boxes.numel() != 0: # Check if boxes need explicit emptying
                          current_boxes = torch.empty((0, 4), device=current_boxes.device, dtype=current_boxes.dtype)
            
                targets.append({
                    "labels": current_labels,  # Might be tensor([])
                    "boxes":  current_boxes    # Might be tensor([], size=(0, 4))
                })

            outputs = {
                "pred_logits": raw_logits,
                "pred_boxes":  pred_boxes
            }
            loss_dict = criterion(outputs, targets)
            weighted_loss_dict = {k: loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict}
            total_loss = sum(weighted_loss_dict.values())
            val_loss += total_loss.item()

            # Custom evaluation. Count the number of occurrences where 
            #1) ground truth class matches with a predicted class and their bounding boxes overlap (to a certain threshold)
            #2) There is a overlapping bounding box and the right class but they aren't the same prediction
            #3) There is a threshold-overlapping bounding box but not the right class
            #4) There is a right class in the prediction but no overlapping box

            total_gt_objects_in_dataset += sum(t['labels'].numel() for t in targets)
            batch_matched_indices = criterion.matcher(outputs, targets) # List of (pred_idx, gt_idx) tuples for each image in batch
            
            for b in range(raw_logits.shape[0]): # Iterate over each image in the batch
                out_logits_img = raw_logits[b]      # Shape: (num_queries, num_classes)
                pred_boxes_img = pred_boxes[b]      # Shape: (num_queries, 4) in cxcywh normalized
                target_img = targets[b]           # Dict: {'labels': (num_gt,), 'boxes': (num_gt, 4)}
                
                # matched_pred_indices_img are indices for predicted queries
                # matched_gt_indices_img are indices for ground truth objects in target_img
                matched_pred_indices_img, matched_gt_indices_img = batch_matched_indices[b]

                num_gt_img = target_img['labels'].shape[0]
                if num_gt_img == 0:
                    continue # No ground truth objects in this image to evaluate against

                num_pred_img = pred_boxes_img.shape[0]
                gt_categorized_flags = [False] * num_gt_img #If image successfully sorted to one of the 4 categories

                # Convert boxes to xyxy for IoU calculation
                pred_boxes_img_xyxy = cxcywh_to_xyxy(pred_boxes_img)
                gt_boxes_img_xyxy = cxcywh_to_xyxy(target_img['boxes'])
                
                pred_classes_img = torch.argmax(out_logits_img, dim=-1) # Shape: (num_queries,)

                # Case 1: Ground truth class matches with a *matched* predicted class and their bounding boxes overlap
                for k in range(len(matched_pred_indices_img)):
                    p_idx = matched_pred_indices_img[k]
                    t_idx = matched_gt_indices_img[k]

                    if gt_categorized_flags[t_idx]:
                        continue

                    # Unsqueeze to make them [1,4] for box_iou
                    iou_matrix = torchvision.ops.box_iou(pred_boxes_img_xyxy[p_idx].unsqueeze(0), 
                                                         gt_boxes_img_xyxy[t_idx].unsqueeze(0))
                    iou = iou_matrix[0, 0]
                    
                    pred_c = pred_classes_img[p_idx]
                    gt_c = target_img['labels'][t_idx]

                    if pred_c == gt_c and iou > iou_threshold:
                        case1_counts += 1
                        gt_categorized_flags[t_idx] = True
                
                # Case 2: There is an overlapping bounding box and the right class but they aren't the same prediction
                for t_idx in range(num_gt_img):
                    if gt_categorized_flags[t_idx]:
                        continue

                    gt_b_xyxy_single = gt_boxes_img_xyxy[t_idx].unsqueeze(0) # Shape [1,4]
                    gt_c = target_img['labels'][t_idx]
                    
                    for p_idx in range(num_pred_img):
                        # Check if this (p_idx, t_idx) pair was THE Hungarian match for t_idx
                        is_p_idx_the_hungarian_match_for_t_idx = False
                        for k_match in range(len(matched_pred_indices_img)):
                            if matched_pred_indices_img[k_match] == p_idx and \
                               matched_gt_indices_img[k_match] == t_idx:
                                is_p_idx_the_hungarian_match_for_t_idx = True
                                break
                        
                        if is_p_idx_the_hungarian_match_for_t_idx:
                            continue # This pred was optimally matched to this GT. If not Case 1, not Case 2 with this pred.

                        pred_b_xyxy_single = pred_boxes_img_xyxy[p_idx].unsqueeze(0) # Shape [1,4]
                        pred_c = pred_classes_img[p_idx]
                        
                        iou_matrix = torchvision.ops.box_iou(pred_b_xyxy_single, gt_b_xyxy_single)
                        iou = iou_matrix[0, 0]

                        if pred_c == gt_c and iou > iou_threshold:
                            case2_counts += 1
                            gt_categorized_flags[t_idx] = True
                            break # This GT object is now categorized for Case 2

                # Case 3: There is a threshold-overlapping bounding box but not the right class
                for t_idx in range(num_gt_img):
                    if gt_categorized_flags[t_idx]:
                        continue

                    gt_b_xyxy_single = gt_boxes_img_xyxy[t_idx].unsqueeze(0)
                    gt_c = target_img['labels'][t_idx] # Needed to check pred_c != gt_c

                    for p_idx in range(num_pred_img):
                        pred_b_xyxy_single = pred_boxes_img_xyxy[p_idx].unsqueeze(0)
                        pred_c = pred_classes_img[p_idx]
                        
                        iou_matrix = torchvision.ops.box_iou(pred_b_xyxy_single, gt_b_xyxy_single)
                        iou = iou_matrix[0, 0]

                        if iou > iou_threshold and pred_c != gt_c:
                            case3_counts += 1
                            gt_categorized_flags[t_idx] = True
                            break # This GT object is now categorized for Case 3
                
                # Case 4: There is a right class in the prediction but no overlapping box (or low overlap)
                for t_idx in range(num_gt_img):
                    if gt_categorized_flags[t_idx]:
                        continue

                    gt_b_xyxy_single = gt_boxes_img_xyxy[t_idx].unsqueeze(0)
                    gt_c = target_img['labels'][t_idx]

                    for p_idx in range(num_pred_img):
                        pred_b_xyxy_single = pred_boxes_img_xyxy[p_idx].unsqueeze(0)
                        pred_c = pred_classes_img[p_idx]
                        
                        iou_matrix = torchvision.ops.box_iou(pred_b_xyxy_single, gt_b_xyxy_single)
                        iou = iou_matrix[0, 0]

                        if pred_c == gt_c and iou <= iou_threshold:
                            case4_counts += 1
                            gt_categorized_flags[t_idx] = True
                            break # This GT object is now categorized for Case 4

                    
    
    # Calculate average loss
    avg_val_loss = val_loss / len(dataloader)
    
    custom_evaluation_results = {
        "case1_gt_match_pred_class_box_overlap": case1_counts,
        "case2_overlap_right_class_not_hungarian_match": case2_counts,
        "case3_overlap_wrong_class": case3_counts,
        "case4_right_class_no_overlap": case4_counts,
        "total_evaluated_gt_objects": total_gt_objects_in_dataset
    }

    return avg_val_loss, custom_evaluation_results


def compute_iou(box1, box2):
    """box1: [N, 4], box2: [M, 4] in xyxy format"""
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N, M, 2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou

def show_image_prediction(
    config,
    model,
    dataset,
    iou,
    file_name,
    images_path,
    with_images = True, #Whether the input to the model is supposed to be images
    all_boxes = True #Whether to output all bounding boxes or only correctly predicted classes and overlapping ones
):
           
    
    # Get image files in archive
    image = Image.open(os.path.join(images_path, file_name)).convert("RGB")
    width, height = image.size

    print("Passing chosen datapoint through model")
    print("######################################")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    device = torch.device(config.device)

    model.eval()
    model.to(device)

    with torch.no_grad():
        for images, data in dataloader:

            #Only pass given file
            if data['file_name'][0] != file_name:
                continue

            if with_images:
                inputs = images.to(device)
            else:
                inputs = data['clip_vision'].to(device)
            raw_logits, raw_pred_boxes = model(inputs)
                
            pred_scores = raw_logits.softmax(-1)[0]  # [num_queries, num_classes]
            pred_labels = pred_scores.argmax(-1)     # [num_queries]
            pred_confs = pred_scores.max(-1).values  # [num_queries]
            pred_boxes = raw_pred_boxes[0]  # [num_queries, 4] in cxcywh              
            pred_boxes_xyxy = cxcywh_to_xyxy(pred_boxes) * torch.tensor([width, height, width, height], device=device)


            # Count most common label predicted and its mean confidence
            label_counter = Counter(pred_labels.tolist())
            most_common_label, count = label_counter.most_common(1)[0]
            confidences_of_label = pred_confs[pred_labels == most_common_label]
            avg_conf = confidences_of_label.mean().item()
            print(f"Most predicted label: {most_common_label} — {count} times, avg confidence: {avg_conf:.2f}")

            true_boxes = convert_boxes(data["bbox"][0].to(device), (width, height))  # [N, 4] cxcywh
            true_boxes_xyxy = cxcywh_to_xyxy(true_boxes) * torch.tensor([width, height, width, height], device=device)
            true_labels = data["labels"][0].to(device)
            if true_labels.dim() == 0:
                true_labels = true_labels.unsqueeze(0)
                
            # Filter predictions based on IoU or correct class
            selected_indices = []
            if all_boxes:
                print("Showing all bounding boxes")
                selected_indices = list(range(len(pred_boxes)))
            else:
                if true_boxes_xyxy.shape[0] > 0:
                    ious = compute_iou(pred_boxes_xyxy, true_boxes_xyxy)  # [num_preds, num_gt]
                    max_ious, _ = ious.max(dim=1)
                else:
                    max_ious = torch.zeros(len(pred_boxes), device=device)

                for idx, (pred_label, iou_val) in enumerate(zip(pred_labels, max_ious)):
                    if pred_label in true_labels or iou_val >= iou:
                        selected_indices.append(idx)

                if len(selected_indices) == 0:
                    if true_boxes_xyxy.shape[0] > 0:
                        ious = compute_iou(pred_boxes_xyxy, true_boxes_xyxy)  # [num_preds, num_gt]
                        max_ious, _ = ious.max(dim=1)  # [num_preds]
                        if (max_ious > 0).any():
                            best_idx = max_ious.argmax().item()
                            print(f"No correct prediction, showing closest box with non-zero IoU = {max_ious[best_idx]:.2f}")
                        else:
                            best_idx = pred_confs.argmax().item()
                            print(f"No overlap with any GT — showing most confident prediction (conf = {pred_confs[best_idx]:.2f})")
                    else:
                        best_idx = pred_confs.argmax().item()
                        print(f"No ground-truth boxes — showing most confident prediction (conf = {pred_confs[best_idx]:.2f})")
                
                    selected_indices = [best_idx]

    # Visualize
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    # Draw predicted boxes
    for idx in selected_indices:
        x1, y1, x2, y2 = pred_boxes_xyxy[idx].cpu()
        label = pred_labels[idx].item()
        conf = pred_confs[idx].item()
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                    linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f"Pred: {label} ({conf:.2f})", color='red',
                fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    # Draw ground-truth boxes
    for idx in range(len(true_boxes_xyxy)):
        x1, y1, x2, y2 = true_boxes_xyxy[idx].cpu()
        label = true_labels[idx].item()
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                    linewidth=2, edgecolor='green', facecolor='none', linestyle='--')
        ax.add_patch(rect)
        ax.text(x1, y2 + 5, f"GT: {label}", color='green',
                fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    plt.axis("off")
    plt.tight_layout()

    savedirectory = "box_images/" + config.train.final_checkpoint_name[:-4]
    if not os.path.exists(savedirectory):
        os.makedirs(savedirectory)

    plt.savefig(+ savedirectory + "/" + file_name, bbox_inches="tight", pad_inches=0.1)
    plt.show()