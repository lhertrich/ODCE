import torch
import wandb
import omegaconf
from hydra.utils import instantiate
from detr.models.matcher import HungarianMatcher
from detr.models.detr    import SetCriterion
from data.dataloader import get_dataloader
import hydra
import os
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

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

class DetectionModel(pl.LightningModule):
    def __init__(self, model, criterion, weight_dict, config):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.weight_dict = weight_dict
        self.config = config

    def forward(self, x):
        return self.model(x)

    def step(self, batch):
        images, data = batch
        inputs = images if self.config.data.load_images else data['clip_vision']
        labels = data["labels"]
        boxes = data["bbox"]
        _, _, h, w = inputs.shape
        img_dims = [(w, h)] * len(boxes)
        converted_boxes = [convert_boxes(box, img_size) for box, img_size in zip(boxes.unbind(0), img_dims)]
        processed_labels = [lbl.unsqueeze(0) if lbl.dim() == 0 else lbl for lbl in labels.unbind(0)]
        targets = [{"labels": lbl, "boxes": box} for lbl, box in zip(processed_labels, converted_boxes)]

        raw_logits, raw_pred_boxes = self(inputs)
        outputs = {
            "pred_logits": raw_logits,
            "pred_boxes": raw_pred_boxes.sigmoid()
        }
        loss_dict = self.criterion(outputs, targets)
        weighted_loss_dict = {k: loss_dict[k] * self.weight_dict[k] for k in loss_dict.keys() if k in self.weight_dict}
        total_loss = sum(weighted_loss_dict.values())
        return total_loss, loss_dict

    def training_step(self, batch, batch_idx):
        total_loss, loss_dict = self.step(batch)
        self.log("train/loss", total_loss, on_step=True, on_epoch=True)
        for k, v in loss_dict.items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss, loss_dict = self.step(batch)
        self.log("val/loss", total_loss, on_epoch=True)
        for k, v in loss_dict.items():
            self.log(f"val/{k}", v, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.config.optimizer.lr)

@hydra.main(version_base=None, config_path='',
            config_name='config')
def main(config):
    wandb_logger = WandbLogger(
    project=config.wandb.project,
    entity=config.wandb.entity,
    name=config.wandb.name,
    config=omegaconf.OmegaConf.to_container(config, resolve=True)
    )
    
    if config.model == 'orgDETR':
        model = instantiate(config.org_detr)
    elif config.model == 'adapterDETR':
        model = instantiate(config.adapter_detr)
    else:
        raise ValueError(f"Unknown model type: {config.model}")

    # Prepare data and components
    train_loader, val_loader = get_dataloader(config)
    criterion, weight_dict = get_loss_fn(config)
    pl_model = DetectionModel(model, criterion, weight_dict, config)

    # Configure checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.train.checkpoint_dir,
        filename="{epoch}-{step}-{val/loss:.2f}",
        save_top_k=3,
        monitor="val/loss",
        mode="min"
    )
    
    batch_size = config.data.batch_size
    print(f"Batch size: {batch_size}")
    print("Training started")
    model = DetectionModel(model, criterion, weight_dict, config)
    
    resume_checkpoint = None
    if os.path.exists(config.train.resume_checkpoint_dir):
        resume_checkpoint = config.train.resume_checkpoint_dir
        print(f"Resuming from checkpoint: {resume_checkpoint}")
    
    checkpoint_callback=ModelCheckpoint(
        dirpath=config.train.checkpoint_dir,
        monitor="val/loss",
        filename="step-{step}",
        every_n_train_steps=config.train.save_interval,
        save_top_k=-1  # keep all
        )
    
    trainer = Trainer(
        default_root_dir=config.train.checkpoint_dir,
        max_epochs=config.train.epochs,
        logger=wandb_logger,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval='step')
        ],
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