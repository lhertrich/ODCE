export PYTHONPATH="$HOME/ODCE/detr/:$PYTHONPATH"
export HYDRA_FULL_ERROR=1
python main.py \
    num_queries=100 \
    train.checkpoint_dir=./outputs/train-freeze-backbone \
    train.resume_checkpoint_dir='"./outputs/train-freeze-backbone/epoch=8-step=10000-val/total_loss=3.31.ckpt"' \
    wandb.name="train-default-queries-num-queries-100"

    
