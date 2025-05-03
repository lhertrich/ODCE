import torch
import wandb
import omegaconf
from hydra.utils import instantiate
from detr.models.matcher import HungarianMatcher
from detr.models.detr    import SetCriterion
from data.dataloader import get_dataloader
import hydra

@hydra.main(version_base=None, config_path='',
            config_name='config')
def main(config):
    # 3. Initialize WandB
    wandb.init(**config.wandb, config=omegaconf.OmegaConf.to_container(config))

    model = instantiate(config.adapter_detr)
    optimizer = instantiate(config.optimizer, params=model.parameters())
    
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
  
    # 5. Training loop
    dataloader = get_dataloader(dataset_name=config.data.dataset_name,
                                feature_path=config.data.feature_path,
                                split=config.data.split,
                                batch_size=config.data.batch_size,
                                shuffle=config.data.shuffle)
    model.train()
    print("Training started")
    for epoch in range(config.train.epochs):
        for images, data in dataloader:
            inputs = data['clip_vision']
            outputs = model.forward(inputs.unsqueeze(1))
            """
            TODO: fix the loss function
            """
            loss = criterion(outputs, data)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log metrics based on training step information not epoch
            wandb.log({
                "epoch": epoch,
                "loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]['lr']
            })

    # Finish WandB run
    wandb.finish()

if __name__ == "__main__":
    main()