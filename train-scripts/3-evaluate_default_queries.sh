export PYTHONPATH="$HOME/ODCE/detr/:$PYTHONPATH"
export HYDRA_FULL_ERROR=1
python testing.py \
    num_queries=100 \
    train.checkpoint_dir=./outputs/train-freeze-backbone \
    train.final_checkpoint_name='"epoch=52-step=60000-val/total_loss=3.92.ckpt"' \
    wandb.name="train-default-queries-num-queries-100"

    
