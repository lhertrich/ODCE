import torch
import wandb
import omegaconf
from hydra.utils import instantiate
from detr.models.matcher import HungarianMatcher
from detr.models.detr    import SetCriterion
from data.dataloader import get_dataloader
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
        num_classes=config.num_classes+1,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=eos_coef,
        losses=['labels','boxes']
    )
    return criterion, weight_dict

@hydra.main(version_base=None, config_path='',
            config_name='config')
def main(config):
    # 3. Initialize WandB
    wandb.init(**config.wandb, config=omegaconf.OmegaConf.to_container(config))
    global_step = 0
    device = torch.device(config.device)
    checkpoint_dir = config.train.checkpoint_dir  
    os.makedirs(checkpoint_dir, exist_ok=True)  
    model = instantiate(config.adapter_detr)
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
    dataloader = get_dataloader(dataset_name=config.data.dataset_name,
                                feature_path=config.data.feature_path,
                                split=config.data.split,
                                batch_size=config.data.batch_size,
                                shuffle=config.data.shuffle, 
                                device=device)    
    model.train()
    criterion, weight_dict = get_loss_fn(config)
    criterion = criterion.to(device)
    
    print("Training started")
    for epoch in range(start_epoch, config.train.epochs):
        for images, data in dataloader:
            inputs = data['clip_vision'].to(device)
            raw_logits, raw_pred_boxes = model.forward(inputs.unsqueeze(1))
            labels = data["labels"].to(device)
            boxes = data["bbox"].to(device)
            converted_boxes = [convert_boxes(box) for box in boxes.unbind(0)]

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
                {"labels": lbl, "boxes": box} 
                for lbl, box in zip(processed_labels, converted_boxes)
            ]

            outputs = {
                "pred_logits": raw_logits,
                "pred_boxes": raw_pred_boxes
            }

            loss_dict = criterion(outputs, targets)
            weighted_loss_dict = {k: loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict}
            total_loss = sum(weighted_loss_dict.values())
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Log both total and individual losses
            log_dict = {f"loss/{k}": v.item() for k, v in loss_dict.items()}
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