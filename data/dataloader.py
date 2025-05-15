from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import torch
from torchvision.transforms import ToTensor
from torchvision import transforms as T

"""
Note: Bbox format is xywh.
"""


class RefL4Dataset(Dataset):
    def __init__(
        self,
        dataset_path,
        split,
        transforms=None,
        load_images=False,
        load_features=True,
        feature_dir=None,
    ):
        super(RefL4Dataset, self).__init__()
        assert split in ["val", "test", "train"], "split should be val or test, train"
        self.dataset_path = dataset_path
        self.split = split
        self.transforms = transforms
        self.feature_dir = Path(feature_dir) if feature_dir else None
        self.load_images = load_images
        self.load_features = load_features
        self.vision_features = None
        self.text_features = None
        self._load_dataset()

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

    def _load_clip_features(self):
        if self.feature_dir is not None:
            self.vision_features = torch.load(
                f"{self.feature_dir}/{self.split}_vision_features.pt"
            )
            self.text_features = torch.load(
                f"{self.feature_dir}/{self.split}_text_features.pt"
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

        # output = {
        #     "bbox": torch.tensor(sample["bbox"], dtype=torch.float32),
        #     "labels": int(sample["ori_category_id"].split("_")[1]) - 1,
        #     "image_file": image_path,
        # }
        output = dict(sample)
        output["bbox"] = torch.tensor(sample["bbox"], dtype=torch.float32)
        output["labels"] = int(sample["ori_category_id"].split("_")[1]) - 1

        if self.load_images:
            image = Image.open(image_path).convert("RGB")
            if self.transforms:
                image = self.transforms(image)
            else:
                image = ToTensor()(image)
            output["image"] = image

        if self.load_features:
            vision_feature = self.vision_features[idx]
            text_feature = self.text_features[idx]
            output["clip_vision"] = vision_feature
            output["clip_text"] = text_feature

        return output


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
