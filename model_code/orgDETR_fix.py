import torch
import torch.nn as nn
from transformers import DetrForObjectDetection, DetrImageProcessor
import types
from typing import Tuple

class Detr(nn.Module):
    def __init__(
        self,
        model_checkpoint: str = "facebook/detr-resnet-50",
        freeze_backbone: bool = True,
        freeze_all: bool = False,
        num_classes: int = 365 + 1  # 365 objects + 1 background
    ):
        super().__init__()
        # Load pretrained model and processor
        self.detr = DetrForObjectDetection.from_pretrained(model_checkpoint)
        self.processor = DetrImageProcessor.from_pretrained(model_checkpoint)
        
        # Replace classification head
        hidden_dim = self.detr.config.d_model
        self.classifier = nn.Linear(hidden_dim, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.detr.model.backbone.parameters():
                param.requires_grad = False
        if freeze_all:
            for param in self.detr.parameters():
                param.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.detr.model(pixel_values).last_hidden_state
        return self.classifier(features), self.detr.bbox_predictor(features)

    @torch.no_grad()
    def predict(self, pixel_values: torch.Tensor, threshold: float = 0.9):
        self.eval()
        logits, boxes = self(pixel_values)
        
        # Post-process detections
        target_sizes = torch.tensor([pixel_values.shape[-2:]] * len(pixel_values))
        outputs = types.SimpleNamespace(logits=logits, pred_boxes=boxes)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=threshold
        )

        # Format outputs
        detections = []
        for result in results:
            img_detections = []
            for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
                img_detections.append({
                    "score": round(score.item(), 3),
                    "label": self.detr.config.id2label.get(label_id.item(), "unknown"),
                    "box": [round(float(c), 2) for c in box]
                })
            detections.append(img_detections)
            
        return detections