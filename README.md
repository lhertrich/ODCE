# ODCE
Project repository for the project object detection from clip embeddings for the EPFL course Visual Intelligence.

Dataset is saved at data/longclip-emb/text_features.pt & data/longclip-emb/vision_features.pt
import detr is a bit tricky, I cloned the repo and added it to the path.
git clone https://github.com/facebookresearch/detr.git

Start an interactive session on a compute node (eg, 2 GPUs case):

```bash
srun -t 120 -A cs-503 --qos=cs-503 --gres=gpu:2 --mem=16G --pty bash
```
Then, on the compute node:

```bash
conda activate nanofm
wandb login
OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 run_training.py 