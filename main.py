import torch
import wandb
import omegaconf
from hydra.utils import instantiate
from detr.models.matcher import HungarianMatcher
from detr.models.detr import SetCriterion
from data.dataloader import get_dataloader
from torch.optim.adamw import AdamW
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F
import hydra
import os


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


def get_loss_fn(config):
    # 1. Initialize the matcher
    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)

    weight_dict = {
        "loss_ce": 1.0,  # classification
        "loss_bbox": 5.0,  # L1
        "loss_giou": 2.0,  # GIoU
    }
    eos_coef = 0.1  # weight for no-object class
    criterion = SetCriterion(
        num_classes=config.num_classes,  # +1,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=eos_coef,
        losses=["labels", "boxes", "cardinality"],
    )
    return criterion, weight_dict


@hydra.main(version_base=None, config_path="", config_name="config")
def main(config):
    wandb.init(**config.wandb, config=omegaconf.OmegaConf.to_container(config))
    global_step = 0
    device = torch.device(config.device)

    # Init model
    if config.model == "org_detr":
        model = instantiate(config.org_detr)
        checkpoint_dir = config.train.org_checkpoint_dir
        resume_checkpoint_dir = config.train.org_resume_checkpoint_dir
    elif config.model == "adapter_detr":
        model = instantiate(config.adapter_detr)
        checkpoint_dir = config.train.checkpoint_dir
        resume_checkpoint_dir = config.train.resume_checkpoint_dir
    else:
        raise ValueError(f"Unknown model type: {config.model}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Init optimizer
    optimizer = AdamW(
        model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay
    )
    last_checkpoint_path = os.path.join(
        checkpoint_dir, config.train.final_checkpoint_name
    )
    resume_from_scratch = config.train.from_scratch

    if os.path.exists(resume_checkpoint_dir) and not resume_from_scratch:
        print(f"Loading model from {checkpoint_dir}")
        checkpoint = torch.load(resume_checkpoint_dir, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
        print(
            f"\nResumed training from {config.train.resume_checkpoint_dir} (epoch {checkpoint['epoch']}, step {global_step})"
        )
    else:
        start_epoch = 0
        print("No checkpoint found, starting from scratch.")

    # Init dataloader
    model.to(device)

    # train_ds, val_ds = torch.utils.data.random_split(
    #     full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    # )
    # train_loader = torch.utils.data.DataLoader(
    #     train_ds, batch_size=config.data.batch_size, shuffle=True
    # )
    # val_loader = torch.utils.data.DataLoader(
    #     val_ds, batch_size=config.data.batch_size, shuffle=False
    # )

    _, train_loader = get_dataloader(config, split="train")
    _, val_loader = get_dataloader(config, split="val")

    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(val_loader.dataset)}")

    # Loss function
    criterion, weight_dict = get_loss_fn(config)
    criterion = criterion.to(device)
    batch_size = config.data.batch_size
    print(f"Batch size: {batch_size}")
    print("Training started")
    model.train()

    # Scheduler
    num_training_steps = config.train.epochs * len(train_loader)
    num_warmup_steps = config.train.warmup_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Training loop
    for epoch in range(start_epoch, config.train.epochs):
        for images, data in train_loader:
            
            if config.data.load_images:
                inputs = images.to(device)
                raw_logits, raw_pred_boxes = model.forward(inputs)
            else:
                inputs = data["clip_vision"].to(device)
                raw_logits, raw_pred_boxes = model.forward(inputs)

            labels = data["labels"].to(device)
            boxes = data["bbox"].to(device)
            widths = data["width"].tolist()
            heights = data["height"].tolist()

            # Normalize & convert each sample’s box to (cx,cy,w,h) in [0,1]
            converted_boxes = [
                convert_boxes(box, (w, h))
                for box, w, h in zip(boxes.unbind(0), widths, heights)
            ]

            processed_labels = []
            for i, lbl in enumerate(labels.unbind(0)):
                # If label is scalar (0-dim), convert to 1D tensor
                if lbl.dim() == 0:
                    lbl = lbl.unsqueeze(0)
                # If label is empty, add no object label (class labels from 0-364)
                # if lbl.numel() == 0:
                #     lbl = torch.tensor([365], device=lbl.device)

                # valid_mask = (lbl >= 0) & (lbl < config.num_classes)
                # if not valid_mask.all():
                #     print(
                #         f"WARNING: Found invalid label(s): {lbl[~valid_mask].tolist()}"
                #     )
                #     # Replace invalid labels with 0 (background)
                #     lbl = torch.where(valid_mask, lbl, torch.zeros_like(lbl))
                processed_labels.append(lbl)

            targets = [
                {
                    "labels": lbl,  # (n_obj,)
                    "boxes": box,
                }  # (n_obj,4), now in cxcywh normalized coords
                for lbl, box in zip(processed_labels, converted_boxes)
            ]

            # DEBUGGING
            # pred_classes = raw_logits.argmax(-1)  # [B, num_queries], ints in [0..num_classes]
            # gt_classes   = [t["labels"] for t in targets]  # list of length B

            # for j in range(len(gt_classes)):
            #     # take only the first e.g. 10 query slots so prints don’t explode
            #     pc = pred_classes[j, :10].tolist()
            #     gc = gt_classes[j].tolist()
            #     print(f"Step {i}Sample {j}  PRED>> {pc}  |  GT>> {gc}")

            outputs = {
                "pred_logits": raw_logits,
                "pred_boxes": raw_pred_boxes,  # Maybe sigmoid?
            }

            loss_dict = criterion(outputs, targets)
            weighted_loss_dict = {
                k: loss_dict[k] * weight_dict[k]
                for k in loss_dict.keys()
                if k in weight_dict
            }
            total_loss = sum(weighted_loss_dict.values())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            # Log both total and individual losses
            log_dict = {f"loss/{k}": v.item() for k, v in loss_dict.items()}
            # print(global_step, log_dict)
            log_dict.update(
                {
                    "epoch": epoch,
                    "loss/total": total_loss.item(),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
            wandb.log(log_dict)

            global_step += 1
            if global_step % config.train.save_interval == 0:
                checkpoint_path = os.path.join(
                    checkpoint_dir, f"checkpoint_step_{global_step}.pth"
                )
                torch.save(
                    {
                        "global_step": global_step,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": total_loss.item(),
                    },
                    checkpoint_path,
                )

        model.eval()
        val_loss = 0.0
        custom_ce_loss = 0.0
        count = 0
        with torch.no_grad():
            for images, data in val_loader:
                if config.data.load_images:
                    inputs = images.to(device)
                    raw_logits, raw_pred_boxes = model.forward(inputs)
                else:
                    inputs = data["clip_vision"].to(device)
                    raw_logits, raw_pred_boxes = model(inputs)

                pred_boxes = raw_pred_boxes.sigmoid()
                labels = data["labels"].to(device)
                boxes = data["bbox"].to(device)  # shape (B,4)
                widths = data["width"].tolist()  # [w1, w2, …]
                heights = data["height"].tolist()  # [h1, h2, …]

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
                    # # If label is empty, add no object label (class labels from 0-364)
                    # if lbl.numel() == 0:
                    #     lbl = torch.tensor([365], device=lbl.device)
                    processed_labels.append(lbl)

                targets = [
                    {
                        "labels": lbl,  # (n_obj,)
                        "boxes": box,
                    }  # (n_obj,4), now in cxcywh normalized coords
                    for lbl, box in zip(processed_labels, converted_boxes)
                ]

                outputs = {"pred_logits": raw_logits, "pred_boxes": pred_boxes}
                loss_dict = criterion(outputs, targets)
                weighted_loss_dict = {
                    k: loss_dict[k] * weight_dict[k]
                    for k in loss_dict.keys()
                    if k in weight_dict
                }
                total_loss = sum(weighted_loss_dict.values())
                val_loss += total_loss.item()
                count += 1

                custom_loss = 0.0
                for i, target in enumerate(targets):
                    if target["labels"].numel() > 0:  # Skip empty targets
                        # Get logits without last column (no-object class)
                        logits_no_obj = outputs["pred_logits"][
                            i, :, :-1
                        ]  # [num_queries, num_classes-1]

                        # For each target label, find the most similar query
                        for label in target["labels"]:
                            # Get probability of the correct class for each query
                            label_probs = F.softmax(logits_no_obj, dim=1)[
                                :, label
                            ]  # [num_queries]
                            # Add negative log likelihood of the best matching query
                            custom_loss -= torch.log(label_probs.max() + 1e-6)

        avg_val_loss = val_loss / count
        avg_custom_ce_loss = custom_ce_loss / count
        # log to wandb
        wandb.log(
            {
                "epoch": epoch,
                "val/loss": avg_val_loss,
                "val/custom_ce_loss": avg_custom_ce_loss,
            }
        )

    final_checkpoint = {
        "global_step": global_step,
        "epoch": config.train.epochs - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": total_loss.item(),
    }
    torch.save(final_checkpoint, last_checkpoint_path)

    wandb.finish()


if __name__ == "__main__":
    main()
