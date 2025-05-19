from pathlib import Path
import json
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import torch
from torchvision.transforms import ToTensor
from torchvision import transforms as T

"""
Note: Bbox format is xywh.
"""

from torch.utils.data import Dataset
from PIL import Image
import tarfile
import io
from tqdm import tqdm
import torch
import datasets
from torchvision.transforms import ToTensor
from pathlib import Path
import logging
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO)


class RefL4Dataset(Dataset):
    def __init__(
        self,
        dataset_path,
        split,
        transforms=None,
        load_images=True,
        load_features=False,
        feature_dir=None,
        cache_dir=None,
    ):
        super(RefL4Dataset, self).__init__()
        assert split in ["val", "test", "train"], "split should be val or test, train"
        self.dataset_path = dataset_path
        self.split = split
        self.transforms = transforms
        self.feature_dir = Path(feature_dir) if feature_dir else None
        self.load_images = load_images
        self.load_features = load_features
        try:
            self.cache_dir = Path(cache_dir)
        except:
            print(f"Cache directory not found: {cache_dir}")
        self.vision_features = None
        self.text_features = None
        self._load_dataset()

        project_root = Path(__file__).parent.absolute()

        with open(os.path.join(project_root, "id2label.json"), "r") as f:
            self.id2label = json.load(f)

        with open(os.path.join(project_root, "label2id.json"), "r") as f:
            self.label2id = json.load(f)

        with open(os.path.join(project_root, "ref2o365.json"), "r") as f:
            self.ref2o365 = json.load(f)

        if load_features:
            if self.feature_dir:
                self._load_clip_features()
            else:
                pass
                # TODO: obtain the features directly from CLIP
            assert (
                self.vision_features is not None and self.text_features is not None
            ), "CLIP features not found in specified directory"

    def _load_dataset(self):
        self.dataset_df = pd.read_parquet(f"{self.dataset_path}/{self.split}.parquet")
        print(f"Loaded {len(self.dataset_df)} samples from {self.split} split")

    def _load_clip_features(self):
        if self.feature_dir is not None:
            self.vision_features = torch.load(
                f"{self.feature_dir}/{self.split}/vision_features.pt", weights_only=True
            )
            self.text_features = torch.load(
                f"{self.feature_dir}/{self.split}/text_features.pt", weights_only=True
            )

        assert len(self.vision_features) == len(
            self.dataset_df
        ), "Mismatch in number of vision features and dataset samples"
        assert len(self.text_features) == len(
            self.dataset_df
        ), "Mismatch in number of text features and dataset samples"

    def __len__(self):
        return len(self.dataset_df)

    def __getitem__(self, idx):
        sample = self.dataset_df.iloc[idx]
        image_path = f"{self.dataset_path}/images/{sample['file_name']}"

        output = dict(sample)
        category_id = self.ref2o365.get(
            output["ori_category_id"], output["ori_category_id"]
        )
        output["labels"] = self.label2id.get(category_id, 0)
        output["bbox"] = torch.tensor(sample["bbox"], dtype=torch.float32)

        if self.load_images:
            image = Image.open(image_path).convert("RGB")
            if self.transforms:
                image = self.transforms(image)

            else:
                image = ToTensor()(image)
            # output["image"] = image
        else:
            image = torch.tensor(0)  # Placeholder if images are not loaded

        if self.load_features:
            vision_feature = self.vision_features[idx]
            text_feature = self.text_features[idx]
            output["clip_vision"] = vision_feature
            output["clip_text"] = text_feature

        return image, output


def get_dataloader(config, split, *args, **kwargs):
    if config.data.load_images:
        normalize = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    else:
        normalize = None

    dataset = RefL4Dataset(
        dataset_path=config.data.dataset_path,
        split=split,
        transforms=normalize,
        load_images=config.data.load_images,
        load_features=config.data.load_features,
        feature_dir=config.data.feature_path,
        cache_dir=config.data.cache_dir,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=config.data.shuffle,
        num_workers=config.data.num_workers,
        pin_memory=True,
        *args,
        **kwargs,
    )

    return dataset, dataloader


if __name__ == "__main__":
    normalize = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    dataset = RefL4Dataset(
        dataset_path="./dataset/refl4",
        split="val",
        transforms=normalize,
        load_images=True,
        load_features=False,
        feature_dir=None,
    )

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for batch in dataloader:
        print(batch)
        break
