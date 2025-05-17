# ODCE Validation

For this section to run, the root directory needs to have the Ref-L4 dataset downloaded

huggingface-cli download --repo-type dataset JierunChen/Ref-L4 --local-dir ./Ref-L4

You can put the resulting images.tar.gz in Ref-L4/images.tar.gz (or put it where you want and change the variable of images location in the notebook)

Before running the notebook in validation_config.yaml specify which model you want to validate in the row:

**model: adapter_detr # [org_DETR, adapter_detr]**

I guess you have the model checkpoints and data in the same folders as me.

After this just click run all in the validation.ipynb and hope for the best. For the Qualitative analysis you need to specify what image file you are analyzing.