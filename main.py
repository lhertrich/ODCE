import torch
import wandb
import omegaconf
from hydra.utils import instantiate
from detr.models.matcher import HungarianMatcher
from detr.models.detr    import SetCriterion
from data.dataloader import get_dataloader
import hydra
import os

def convert_boxes_to_cxcywh(boxes, image_size):
    """
    boxes: Tensor[N,4] in (x_min, y_min, x_max, y_max), absolute pixels
    image_size: (width, height)
    returns: Tensor[N,4] in (cx, cy, w, h), normalized to [0,1]
    """
    if boxes.dim() == 1:
        boxes = boxes.unsqueeze(0)

    width, height = image_size
    x_min, y_min, x_max, y_max = boxes.unbind(1)
    w = x_max - x_min
    h = y_max - y_min
    cx = x_min + 0.5 * w
    cy = y_min + 0.5 * h

    # normalize
    cx = cx / width
    cy = cy / height
    w  = w  / width
    h  = h  / height

    return torch.stack([cx, cy, w, h], dim=1)

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

@hydra.main(version_base=None, config_path='',
            config_name='config')
def main(config):
    wandb.init(**config.wandb, config=omegaconf.OmegaConf.to_container(config))
    global_step = 0
    device = torch.device(config.device)
    checkpoint_dir = config.train.checkpoint_dir  
    os.makedirs(checkpoint_dir, exist_ok=True)
    if config.model=='orgDETR':
        model = instantiate(config.org_detr)
    elif config.model=='adapterDETR':  
        model = instantiate(config.adapter_detr)
    else:
        raise ValueError(f"Unknown model type: {config.model}")
    
    optimizer = instantiate(config.optimizer, params=model.parameters())
    last_checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_last.pth")
    
    if os.path.exists(config.train.resume_checkpoint_dir):
        checkpoint = torch.load(config.train.resume_checkpoint_dir, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
        global_step = checkpoint['global_step']
        print(f"\nResumed training from {config.train.resume_checkpoint_dir} (epoch {checkpoint['epoch']}, step {global_step})")
    else: 
        start_epoch = 0
        print("No checkpoint found, starting from scratch.")
    
    model.to(device)
    full_ds, _ = get_dataloader(config)    
    n_total = len(full_ds)
    n_train = int(n_total * 0.9)
    n_val = n_total - n_train
    print('Full dataset length:', n_total, 'Train dataset length:', n_train, 'Validation dataset length:', n_val)
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=config.data.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=config.data.batch_size, shuffle=False)
    
    criterion, weight_dict = get_loss_fn(config)
    criterion = criterion.to(device)
    
    print("Training started")
    model.train()
    for epoch in range(start_epoch, config.train.epochs): 
        for images, data in train_loader:
            if config.data.load_images:
                inputs = images.to(device)
                raw_logits, raw_pred_boxes = model.forward(inputs)
            else:
                inputs = data['clip_vision'].to(device)
                raw_logits, raw_pred_boxes = model.forward(inputs.unsqueeze(1))
            
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
                "pred_boxes": raw_pred_boxes.sigmoid()
            }

            loss_dict = criterion(outputs, targets)
            weighted_loss_dict = {k: loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict}
            total_loss = sum(weighted_loss_dict.values())
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Log both total and individual losses
            log_dict = {f"loss/{k}": v.item() for k, v in loss_dict.items()}
            # print(global_step, log_dict)
            log_dict.update({
                "epoch": epoch,
                "loss/total": total_loss.item(),
                "learning_rate": optimizer.param_groups[0]['lr']
            })
            wandb.log(log_dict)
            
            global_step += 1
            if global_step % config.train.save_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{global_step}.pth")
                torch.save({
                    'global_step': global_step,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': total_loss.item(),
                    }, checkpoint_path)
                
        model.eval()
        val_loss = 0.0
        count = 0
        with torch.no_grad():
            for images, data in val_loader:
                if config.data.load_images:
                    inputs = images.to(device)
                    raw_logits, raw_pred_boxes = model.forward(inputs)
                else:
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
                count    += 1

        avg_val_loss = val_loss / count
        # log to wandb
        wandb.log({
            'epoch': epoch,
            'val/loss': avg_val_loss
        })
                
    final_checkpoint = {
        'global_step': global_step,
        'epoch': config.train.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss.item(),
        }
    torch.save(final_checkpoint, last_checkpoint_path)

    wandb.finish()

if __name__ == "__main__":
    main()