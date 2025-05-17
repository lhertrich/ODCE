from torch.utils.data import Dataset
from PIL import Image
import tarfile
import io
from tqdm import tqdm
import torch
import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision import transforms as T
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
import datasets
import logging
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader, random_split

logging.basicConfig(level=logging.INFO)

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
        images_file="images.tar.gz",
        custom_transforms=None,
        load_images=False,
        load_features=True,
        feature_dir=None,
        cache_dir="./data/cache",
    ):
        super(RefL4Dataset, self).__init__()
        assert split in ["val", "test", "all"], "split should be val, test, or all"
        self.dataset_path = dataset_path
        self.split = split
        self.images_file = images_file
        self.transforms = custom_transforms
        self.feature_dir = Path(feature_dir) if feature_dir else None
        self.load_images = load_images
        self.load_features = load_features
        self.cache_dir = Path(cache_dir)
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
        self.dataset = datasets.load_dataset(self.dataset_path)
        if self.load_images:
            self.images_dir = self.cache_dir / "images"
            if not self.images_dir.exists():
                self.images_dir.mkdir(parents=True, exist_ok=True)
                image_tar_path = None
                if self.dataset_path == "JierunChen/Ref-L4":
                    image_tar_path = hf_hub_download(
                        repo_id="JierunChen/Ref-L4",
                        filename="images.tar.gz",
                        repo_type="dataset",
                    )
                    logging.info(f"Downloaded images.tar.gz to {image_tar_path}")
                else:
                    image_tar_path = Path(self.dataset_path) / self.images_file
                
                self._extract_tar(image_tar_path, self.images_dir)
        
        all_splits = datasets.concatenate_datasets([self.dataset["val"], self.dataset["test"]])
        self.dataset["all"] = all_splits

    def _extract_tar(self, tar_path, extract_dir):
        logging.info(f"Extracting images from {tar_path} to {extract_dir}")
        try:
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=extract_dir)
        except Exception as e:
            logging.error(f"Failed to extract tar file: {str(e)}")
            raise

    def _load_clip_features(self):
        try:
            vision_path = self.feature_dir / "vision_features.pt"
            text_path = self.feature_dir / "text_features.pt"
            self.vision_features = torch.load(vision_path)
            self.text_features = torch.load(text_path)

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
        image = torch.tensor(0)  # Placeholder

        if self.load_images:
            try:
                image_path = self.images_dir / data["file_name"]
                image = Image.open(image_path)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                if self.transforms:
                    image = self.transforms(image)
                else:
                    image = ToTensor()(image)
            except Exception as e:
                logging.error(f"Error loading image {image_path}: {str(e)}")
                raise

        if self.vision_features is not None and self.text_features is not None:
            data["clip_vision"] = self.vision_features[idx]
            data["clip_text"] = self.text_features[idx]

        data["labels"] = int(data["ori_category_id"].split("_")[1]) - 1
        data["bbox"] = torch.tensor(data["bbox"], dtype=torch.float32)

        return image, data

    def change_split(self, split):
        assert split in ["val", "test"], "Invalid split"
        self.split = split
        if self.load_features and self.feature_dir:
            self._load_clip_features()
            
def get_dataloader(config):
    """
    Load saved features and create DataLoader.
    If config.data.split == 'trainval', split the dataset into train and validation sets manually.
    
    Returns:
        tuple: train_loader, val_loader (if split), otherwise just the single dataloader
    """
    if config.data.load_images:
        normalize = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        normalize = None

    dataset = RefL4Dataset(
        dataset_path=config.data.dataset_name,
        split=config.data.split,
        load_images=config.data.load_images,
        load_features=config.data.load_features,
        feature_dir=config.data.feature_path,
        custom_transforms=normalize,
    )

    n_total = len(dataset)
    n_train = int(n_total * 0.8)
    n_val = n_total - n_train
    print('Full dataset length:', n_total, 'Train dataset length:', n_train, 'Validation dataset length:', n_val)

    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=config.data.batch_size, shuffle=True, num_workers=config.data.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.data.batch_size, shuffle=False, num_workers=config.data.num_workers, pin_memory=True)

    return train_loader, val_loader
    # else:
    #     dataloader = DataLoader(
    #         dataset,
    #         batch_size=config.data.batch_size,
    #         shuffle=config.data.shuffle,
    #         num_workers=config.data.num_workers,
    #         pin_memory=True,
    #     )
    #     return dataloader

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
