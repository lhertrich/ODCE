from pathlib import Path
import clip
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger
from dataset import RefL4Dataset
from Long_CLIP.model import longclip


def get_clip_features(
    dataset,
    batch_size,
    save_path,
    model_path,
    device="cuda",
):
    # model, preprocess = clip.load(model_path)
    model, preprocess = longclip.load(model_path, device=device)
    model.float()
    model.to(device)

    dataset.transforms = preprocess
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=RefL4Dataset.collate_fn,
    )

    save_path_vision = Path(f"{save_path}/vision_features.pt")
    save_path_text = Path(f"{save_path}/text_features.pt")

    vision_features = []
    language_features = []
    if save_path_vision.exists():
        vision_features = torch.load(save_path_vision)
        logger.info(f"Load vision features at {save_path_vision}")
    else:
        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(dataloader)):
                images, others = batch
                captions = [x["caption"] for x in others]

                # Get vision features
                images = images.to(device)
                vision_output = model.encode_image(images)
                vision_features.append(vision_output.cpu())

                # Tokenize captions and get language features
                token_inputs = longclip.tokenize(captions, truncate=True).to(device)
                language_output = model.encode_text(token_inputs)
                language_features.append(language_output.cpu())

        vision_features = torch.cat(vision_features, dim=0)
        language_features = torch.cat(language_features, dim=0)

        save_path_vision.parent.mkdir(parents=True, exist_ok=True)
        torch.save(vision_features, save_path_vision)
        logger.info(f"Save vision features at {save_path_vision}")

        save_path_text.parent.mkdir(parents=True, exist_ok=True)
        torch.save(language_features, save_path_text)
        logger.info(f"Save text features at {save_path_text}")

        print("Vision_features.shape", vision_features.shape)
        print("Language_features.shape", language_features.shape)

        return vision_features, language_features


if __name__ == "__main__":
    # for split in ["val", "train"]:
    model_path = "./Long_CLIP/checkpoints/longclip-B.pt"
    for split in ["all"]:
        # Load the dataset
        ref_l4_dataset = RefL4Dataset("JierunChen/Ref-L4", split=split)
        save_path = f"./features/ref_l4/{split}"
        vision_features, language_features = get_clip_features(
            ref_l4_dataset,
            batch_size=32,
            save_path=save_path,
            model_path=model_path,
        )
