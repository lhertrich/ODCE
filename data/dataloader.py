from torch.utils.data import Dataset
from PIL import Image
import tarfile
import io
from tqdm import tqdm
import torch
import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import torch
from torch.utils.data import Dataset
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from pathlib import Path
import tarfile
from PIL import Image
import io
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
import logging

logging.basicConfig(level=logging.INFO)


class RefL4Dataset(Dataset):
    def __init__(
        self,
        dataset_path,
        split,
        images_file="images.tar.gz",
        custom_transforms=None,
        load_images=False,
        load_features=True,
        feature_dir=None,
    ):
        super().__init__()
        assert split in ["val", "test"], "split should be val or test"
        self.dataset_path = dataset_path
        self.split = split
        self.images_file = images_file
        self.transforms = custom_transforms
        self.feature_dir = Path(feature_dir) if feature_dir else None
        self.load_images = load_images
        self.load_features = load_features
        self.vision_features = None
        self.text_features = None

        self._load_dataset()

        if load_features and self.feature_dir:
            self._load_clip_features()
            assert (
                self.vision_features is not None and self.text_features is not None
            ), "CLIP features not found in specified directory"

    def __len__(self):
        return len(self.dataset[self.split])

    def _load_dataset(self):
        self.dataset = load_dataset(self.dataset_path)
        if self.load_images:
            self.images = self._load_images_from_tar(
                f"{self.dataset_path}/{self.images_file}"
            )
        all_splits = concatenate_datasets([self.dataset["val"], self.dataset["test"]])
        self.dataset["all"] = all_splits

    def _load_images_from_tar(self, image_tar_path):
        images = {}
        logging.info(f"Loading images from {image_tar_path}")
        try:
            with tarfile.open(image_tar_path, "r:gz") as tar:
                for member in tqdm(tar.getmembers()):
                    if member.isfile() and member.name.lower().endswith(
                        ("jpg", "jpeg", "png", "webp")
                    ):
                        f = tar.extractfile(member)
                        if f:
                            try:
                                image = Image.open(io.BytesIO(f.read()))
                                if image.mode != "RGB":
                                    image = image.convert("RGB")
                                images[member.name] = image
                            except Exception as e:
                                logging.warning(
                                    f"Failed to load {member.name}: {str(e)}"
                                )
        except Exception as e:
            logging.error(f"Failed to open tar file: {str(e)}")
            raise
        return images

    def _load_clip_features(self):
        try:
            vision_path = self.feature_dir / "vision_features.pt"
            text_path = self.feature_dir / "text_features.pt"
            self.vision_features = torch.load(vision_path)
            self.text_features = torch.load(text_path)

            # Split features based on dataset split
            val_len = len(self.dataset["val"])
            if self.split == "val":
                self.vision_features = self.vision_features[:val_len]
                self.text_features = self.text_features[:val_len]
            elif self.split == "test":
                self.vision_features = self.vision_features[val_len:]
                self.text_features = self.text_features[val_len:]

            assert len(self.vision_features) == len(
                self.dataset[self.split]
            ), "CLIP features length mismatch with dataset"

        except Exception as e:
            logging.error(f"Failed to load CLIP features: {str(e)}")
            raise

    def __getitem__(self, idx):
        data = self.dataset[self.split][idx].copy()
        image = torch.tensor(0)  # Default placeholder

        if self.load_images:
            try:
                image = self.images[data["file_name"]]
                if self.transforms:
                    image = self.transforms(image)
                else:
                    image = ToTensor()(image)
            except KeyError:
                logging.warning(f"Image {data['file_name']} not found in tar")
            except Exception as e:
                logging.error(f"Error processing image {data['file_name']}: {str(e)}")
                raise

        if self.vision_features is not None and self.text_features is not None:
            data["clip_vision"] = self.vision_features[idx]
            data["clip_text"] = self.text_features[idx]

        data["labels"] = (
            int(data["ori_category_id"].split("_")[1]) - 1
        )  # -1 to start from 0
        data["bbox"] = torch.tensor(data["bbox"], dtype=torch.float32)

        return image, data

    def change_split(self, split):
        assert split in ["val", "test"], "Invalid split"
        self.split = split
        if self.load_features and self.feature_dir:
            self._load_clip_features()


# class CLIPFeatureDataset(Dataset):
#     def __init__(self, feature_dir):
#         """
#         Load precomputed CLIP features from directory
#         Args:
#             feature_dir: Path containing vision_features.pt and text_features.pt
#         """
#         self.feature_dir = Path(feature_dir)
#         self.vision_features = torch.load(self.feature_dir / "vision_features.pt")
#         self.text_features = torch.load(self.feature_dir / "text_features.pt")

#         # Verify feature alignment
#         assert len(self.vision_features) == len(self.text_features), \
#             "Mismatch between number of vision and text features"

#     def __len__(self):
#         return len(self.vision_features)

#     def __getitem__(self, idx):
#         return {
#             "vision": self.vision_features[idx],
#             "text": self.text_features[idx]
#         }

#     @property
#     def vision_dim(self):
#         return self.vision_features.shape[1]

#     @property
#     def text_dim(self):
#         return self.text_features.shape[1]


# Usage example
def get_dataloader(
    dataset_name, feature_path, split="val", batch_size=32, shuffle=False, device="cuda"
):
    """
    Load saved features and create DataLoader
    Returns:
        dataset: CLIPFeatureDataset instance
        dataloader: DataLoader for the dataset
    """

    dataset = RefL4Dataset(
        dataset_path=dataset_name,
        split=split,
        load_images=False,
        load_features=True,
        feature_dir=feature_path,
    )

    # Convert string labels to integers in one step

    # def collate_to_device(batch):
    #     return {
    #         key: torch.stack([item[key] for item in batch]).to(device)
    #         for key in batch[0].keys()
    #     }

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        # collate_fn=collate_to_device,
        pin_memory=True,
    )
    return dataloader


if __name__ == "__main__":
    # How to use
    feature_path = "./data/longclip-emb"  # Same path used in get_clip_features()
    dataloader = get_dataloader(
        dataset_name="JierunChen/Ref-L4", feature_path=feature_path
    )

    labels = set()

    # Iterate through batches, note that example (image, data)
    for images, data in dataloader:
        vision_features = data["clip_vision"]
        text_features = data["clip_text"]

        labels.update(data["labels"].tolist())

    print("Num unique labels:", len(labels))
