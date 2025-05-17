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
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger


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
        losses=["labels", "boxes"],
    )
    return criterion, weight_dict

class DetectionModel(pl.LightningModule):
    def __init__(self, model, criterion, weight_dict, config):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.weight_dict = weight_dict
        self.config = config
        self.save_hyperparameters(ignore=['model', 'criterion'])

    def forward(self, x):
        return self.model(x)

    def process_batch(self, batch):
        images, data = batch
        if self.config.data.load_images:
            inputs = images.to(self.device)
            raw_logits, raw_pred_boxes = self.forward(inputs)
            else:
            inputs = data['clip_vision'].to(self.device)
            raw_logits, raw_pred_boxes = self.forward(inputs)

        labels = data["labels"].to(self.device)
        boxes = data["bbox"].to(self.device)
        widths = data["width"].tolist()
        heights = data["height"].tolist()

        # Normalize & convert boxes
            converted_boxes = [
                convert_boxes(box, (w, h)) 
                for box, w, h in zip(boxes.unbind(0), widths, heights)
            ]

            processed_labels = []
            for i, lbl in enumerate(labels.unbind(0)):
                # If label is scalar (0-dim), convert to 1D tensor
                if lbl.dim() == 0:
                    lbl = lbl.unsqueeze(0)
            # If label is empty, add no object label
                if lbl.numel() == 0:
                    lbl = torch.tensor([365], device=lbl.device)

            valid_mask = (lbl >= 0) & (lbl < self.config.num_classes)
                if not valid_mask.all():
                print(f"WARNING: Found invalid label(s): {lbl[~valid_mask].tolist()}")
                    lbl = torch.where(valid_mask, lbl, torch.zeros_like(lbl))
                processed_labels.append(lbl)

            targets = [
            {"labels": lbl, "boxes": box}
            for lbl, box in zip(processed_labels, converted_boxes)
            ]
 
            outputs = {
                "pred_logits": raw_logits,
                "pred_boxes": raw_pred_boxes.sigmoid()
            }

        return outputs, targets

    def training_step(self, batch, batch_idx):
        outputs, targets = self.process_batch(batch)
        loss_dict = self.criterion(outputs, targets)
        weighted_loss_dict = {
            k: loss_dict[k] * self.weight_dict[k]
            for k in loss_dict.keys()
            if k in self.weight_dict
        }
            total_loss = sum(weighted_loss_dict.values())
            
        # Log losses
        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True)
        for k, v in loss_dict.items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        outputs, targets = self.process_batch(batch)
        loss_dict = self.criterion(outputs, targets)
        weighted_loss_dict = {
            k: loss_dict[k] * self.weight_dict[k]
            for k in loss_dict.keys()
            if k in self.weight_dict
        }
        total_loss = sum(weighted_loss_dict.values())

        # Calculate custom CE loss
        custom_loss = 0.0
        for i, target in enumerate(targets):
            if target["labels"].numel() > 0:  # Skip empty targets
                # Get logits without last column (no-object class)
                logits_no_obj = outputs["pred_logits"][i, :, :-1]  # [num_queries, num_classes-1]

                # For each target label, find the most similar query
                for label in target["labels"]:
                    # Get probability of the correct class for each query
                    label_probs = F.softmax(logits_no_obj, dim=1)[:, label]  # [num_queries]
                    # Add negative log likelihood of the best matching query
                    custom_loss -= torch.log(label_probs.max() + 1e-6)

        # Log all losses
        self.log("val/total_loss", total_loss, on_epoch=True)
        self.log("val/custom_ce_loss", custom_loss, on_epoch=True)
        for k, v in loss_dict.items():
            self.log(f"val/{k}", v, on_epoch=True)

        return total_loss

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.config.train.lr,
            weight_decay=self.config.train.weight_decay
        )

        # Setup scheduler with warmup
        num_training_steps = self.trainer.max_epochs * len(self.trainer.train_dataloader)
        num_warmup_steps = self.config.train.warmup_steps

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }

@hydra.main(version_base=None, config_path='',
            config_name='config')
def main(config):
    # Initialize wandb logger
    wandb_logger = WandbLogger(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=config.wandb.name,
        config=omegaconf.OmegaConf.to_container(config, resolve=True)
    )
    
    # Initialize model
    if config.model == 'orgDETR':
        model = instantiate(config.org_detr)
        checkpoint_dir = config.train.org_checkpoint_dir
    elif config.model == 'adapterDETR':
        model = instantiate(config.adapter_detr)
        checkpoint_dir = config.train.checkpoint_dir
    else:
        raise ValueError(f"Unknown model type: {config.model}")

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Prepare data and components
    train_loader, val_loader = get_dataloader(config)
    criterion, weight_dict = get_loss_fn(config)
    pl_model = DetectionModel(model, criterion, weight_dict, config)

    # Configure checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch}-{step}-{val/total_loss:.2f}",
        monitor="val/total_loss",
        save_top_k=3,
        mode="min",
        every_n_train_steps=config.train.save_interval,
    )

    # Configure learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Check for resume checkpoint
    resume_checkpoint = None
    if os.path.exists(config.train.resume_checkpoint_dir):
        resume_checkpoint = config.train.resume_checkpoint_dir
        print(f"Resuming from checkpoint: {resume_checkpoint}")

    # Initialize trainer
    trainer = Trainer(
        default_root_dir=checkpoint_dir,
        max_epochs=config.train.epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        enable_checkpointing=True,
        accelerator="auto",
        devices=config.train.num_devices,
        log_every_n_steps=10,
    )

    # Start training
    trainer.fit(pl_model, train_loader, val_loader, ckpt_path=resume_checkpoint)
    wandb.finish()


if __name__ == "__main__":
    main()
