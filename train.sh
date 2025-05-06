# 1 gpu only, TODO: add multi gpu
export HYDRA_FULL_ERROR=1
export PYTHONPATH="$HOME/ODCE/detr/:$PYTHONPATH"
export PYTHONPATH="$HOME/ODCE/:$PYTHONPATH"
export WANDB_INSECURE_DISABLE_SSL=true
python main.py