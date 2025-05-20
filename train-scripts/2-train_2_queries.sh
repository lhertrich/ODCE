export PYTHONPATH="$HOME/ODCE/detr/:$PYTHONPATH"
export HYDRA_FULL_ERROR=1
python main.py \
    num_queries=2 \
    train.checkpoint_dir=./outputs/train-freeze-backbone-num-queries-2 \
    train.resume_checkpoint_dir='"./outputs/train-freeze-backbone-num-queries-2/epoch=17-step=20000-val/total_loss=4.54.ckpt"' \
    wandb.name="train-2-queries"


    
