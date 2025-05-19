import os
from hydra.utils import instantiate
import hydra
import torch
import wandb
import omegaconf
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from data.dataloader import get_dataloader
from collections import Counter



def load_model(model_config, config, device):
    model = instantiate(model_config)
    checkpoint_path = os.path.join(
        config.checkpoint_dir,
        config.final_checkpoint_name
    )
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
    else:
        print("No checkpoint found, using default weights.")
    return model.to(device).eval()


def convert_boxes(boxes, image_size):
    # boxes: [x_top, y_top, w, h] in pixel coords
    # image_size: (W, H)
    if boxes.dim() == 1:
        boxes = boxes.unsqueeze(0)
    W, H = image_size
    b = boxes.clone().float()
    b[:, 0] /= W
    b[:, 1] /= H
    b[:, 2] /= W
    b[:, 3] /= H
    # now convert top-left→center
    cx = b[:, 0] + 0.5 * b[:, 2]
    cy = b[:, 1] + 0.5 * b[:, 3]
    return torch.stack([cx, cy, b[:, 2], b[:, 3]], dim=1)  # normalized cx,cy,w,h


def compute_iou(box1, box2):
    # both box1, box2 are (cx,cy,w,h) normalized
    def to_xyxy(b):
        cx, cy, w, h = b
        return (cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h)

    x1, y1, x2, y2 = to_xyxy(box1)
    x1g, y1g, x2g, y2g = to_xyxy(box2)

    inter_w = max(0.0, min(x2, x2g) - max(x1, x1g))
    inter_h = max(0.0, min(y2, y2g) - max(y1, y1g))

    inter   = inter_w * inter_h
    area1   = (x2 - x1) * (y2 - y1)
    area2   = (x2g- x1g)* (y2g- y1g)
    union   = area1 + area2 - inter
    return inter/union if union > 0 else 0.0

def make_wandb_box(box, cls, class_labels):
    cx, cy, w, h = box.tolist()
    return {
        "position": {
            "minX": cx - 0.5 * w,
            "minY": cy - 0.5 * h,
            "maxX": cx + 0.5 * w,
            "maxY": cy + 0.5 * h,
        },
        "class_id": int(cls),
    }


@hydra.main(config_path="", config_name="config", version_base=None)
def main(cfg):
    wandb.init(**cfg.test.wandb,
           config=omegaconf.OmegaConf.to_container(cfg.test, resolve=True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg.test.model, cfg.test, device)
    print("Loaded model")
    _, test_loader = get_dataloader(cfg.test, split="test")
    print("Loaded test loader with length", len(test_loader))


    all_preds, all_labels, all_ious = [], [], []

    print("Starting testing")
    with torch.no_grad():
        for images, data in test_loader:
            if cfg.test.data.load_images:
                inputs = images.to(device)
            else:
                inputs = data["clip_vision"].to(device)

            # model forward
            logits, raw_boxes = model(inputs)
            pred_boxes = raw_boxes.cpu()
            pred_logits = logits.cpu()

            gt_labels = data["labels"].cpu()

            gt_raw = data["bbox"].cpu()
            widths = data["width"]
            heights = data["height"]

            B, Q, _ = pred_logits.shape
            for i in range(B):
                scores = torch.softmax(pred_logits[i, :, :-1], dim=-1)
                best_per_query, best_class_per_query = scores.max(dim=-1)
                best_q = best_per_query.argmax().item()

                pred_cls = best_class_per_query[best_q].item()
                pred_box = pred_boxes[i, best_q]

                norm_gt = convert_boxes(
                    gt_raw[i], (widths[i], heights[i])
                )[0]

                gt_box = norm_gt

                iou = compute_iou(pred_box, gt_box)

                all_preds.append(pred_cls)
                all_labels.append(int(gt_labels[i].item()))
                all_ious.append(iou)

    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average=None,
        labels=list(range(cfg.num_classes)),
        zero_division=0,
    )

    wandb.run.summary["test/accuracy"] = acc
    wandb.run.summary["test/mean_IoU"] = sum(all_ious) / len(all_ious)
    wandb.run.summary["test/recall@0.5"] = sum(i >= 0.5 for i in all_ious) / len(
        all_ious
    )

    # Count how many times each class appeared and was predicted
    gt_class_counts = Counter(all_labels)
    pred_class_counts = Counter(all_preds)

    table_columns = [
        "class_id", "precision", "recall", "f1_score", 
        "gt_count", "pred_count"
    ]
    table = wandb.Table(columns=table_columns)
    
    for c in range(cfg.num_classes):
        gt_count_for_class_c = gt_class_counts.get(c, 0)
        pred_count_for_class_c = pred_class_counts.get(c, 0)
        table.add_data(c, p[c], r[c], f1[c], gt_count_for_class_c, pred_count_for_class_c)
    
    wandb.log({"test/per_class_metrics": table})

if __name__ == "__main__":
    main()
