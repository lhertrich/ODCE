import os
from hydra.utils import instantiate
from PIL import Image, ImageDraw, ImageFont
import hydra
import torch
from data.dataloader import get_dataloader


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


@hydra.main(version_base=None, config_path="", config_name="config")
def main(config):
    threshold = 0.5
    image = Image.open("")
    image_feature = torch.load(
        "./ODCE/clip_features/features/ref_l4/val/vision_features.pt",
        weights_only=True,
    )
    image_feature = image_feature[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = instantiate(config.adapter_detr)
    last_checkpoint_path = os.path.join(
        config.train.checkpoint_dir, "checkpoint_last.pth"
    )
    if os.path.exists(last_checkpoint_path):
        print(f"Loading checkpoint from {last_checkpoint_path}")
        checkpoint = torch.load(last_checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("No checkpoint found, using default model weights.")

    inference(
        image=image,
        image_feature=image_feature,
        ground_truth_box=[
            135.56219481850002,
            194.3108520448,
            74.94299316569999,
            114.73577881600002,
        ],
        model=model,
        device=device,
        threshold=threshold,
        output_path="test.png",
    )


if __name__ == "__main__":
    main()
