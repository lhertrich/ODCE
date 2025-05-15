import torch
import torch.nn as nn
from typing import Tuple
import math
import torch.nn.functional as F
import os
from pathlib import Path
from types import SimpleNamespace
import json

#os.environ["TORCH_HOME"] = str(Path("../detr").absolute())

def get_sinusoidal_positional_embedding(batch_size, seq_len, hidden_dim, device):
    position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(
        1
    )
    div_term = torch.exp(
        torch.arange(0, hidden_dim, 2, dtype=torch.float32, device=device)
        * (-math.log(10000.0) / hidden_dim)
    )

    pe = torch.zeros(seq_len, hidden_dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    pe = pe.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, seq_len, hidden_dim)
    return pe

# Helper function for box conversion
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
        (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


class AdapterDetr(nn.Module):

    def __init__(
        self,
        adapter: nn.Module,
        repo_dir: str = "facebookresearch/detr",
        model_name: str = "detr_resnet50",
        train_decoder: bool = True,  
        train_class_head: bool = True,  
        train_bbox_head: bool = True,
        encoder: nn.Module = None,
    ):

        super().__init__()
        self.adapter = adapter
        self.encoder = encoder

        print(f"Loading model and processor from {repo_dir}...")
        self.detr_full_model = torch.hub.load(repo_dir, model_name, pretrained=True)

        project_root = Path(__file__).parent.parent.absolute()

        with open(os.path.join(project_root, "data", "id2label.json"), "r") as f:
            self.id2label = json.load(f)

        with open(os.path.join(project_root, "data", "label2id.json"), "r") as f:
            self.label2id = json.load(f)

        self.config = SimpleNamespace()
        self.config.d_model = 256
        self.config.num_classes = 365
        self.config.id2label = self.id2label
        self.config.label2id = self.label2id
        
        # Get decoder out of DETR model
        self.decoder = self.detr_full_model.transformer.decoder

        # Initialize new prediction head
        num_user_labels = 365
        num_model_classes = num_user_labels + 1
        hidden_dim = self.detr_full_model.class_embed.weight.shape[1]
        self.classification_head = nn.Linear(hidden_dim, num_model_classes)
        nn.init.xavier_uniform_(self.classification_head.weight)

        # FRESH LABEL MAPS IF NECESSARY: self.config.id2label, self.config.label2id

        # Keep bounding box head
        self.bbox_regression_head = self.detr_full_model.bbox_embed

        # Learnable object query positional embeddings (part of DETR)
        # NB! don't understand
        self.query_pos_embed_weight = self.detr_full_model.query_embed.weight
        self.num_queries = self.query_pos_embed_weight.shape[0]
        self.hidden_dim = self.query_pos_embed_weight.shape[1]

        # classification and bbox heads out of DETR
        # self.classification_head = self.detr_full_model.class_labels_classifier
        # self.bbox_regression_head = self.detr_full_model.bbox_predictor

        # Freeze the DETR decoder and query embeddings
        if not train_decoder:
            # Need to freeze decoder parameters directly since there's no "model" attribute
            for param in self.decoder.parameters():
                param.requires_grad = False
            # Also freeze query embeddings if you want
            self.detr_full_model.query_embed.weight.requires_grad = False
            print("Froze DETR decoder parameters")
        else:
            print("DETR decoder trainable")

        # Freeze prediction heads if specified
        if not train_class_head:
            for param in self.classification_head.parameters():
                param.requires_grad = False
            print("DETR class head frozen.")
        else:
            print("DETR class head is trainable.")

        if not train_bbox_head:
            for param in self.bbox_regression_head.parameters():
                param.requires_grad = False
            print("DETR bbox head frozen.")
        else:
            print("DETR bbox head is trainable.")

        print("AdapterDetrModel initialized.")

    def forward(
        self,
        input: torch.Tensor,
        encoder_pos_embed: torch.Tensor = None,
        initial_query_embeddings: torch.Tensor = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        # If the input is given before encoding, encode it
        if self.encoder is not None:
            input = self.encoder(input)
        
        # Pass the encodings through the adapter
        transformed_encoder_hidden_states = self.adapter(input)
        
        batch_size, transformed_seq_len, hidden_dim = transformed_encoder_hidden_states.shape
        
        # PyTorch Hub transformer expects: src = (seq_len, batch_size, hidden_dim)
        memory = transformed_encoder_hidden_states.permute(1, 0, 2)
        
        # Get positional embeddings for the memory
        if encoder_pos_embed is None:
            # The transformer expects pos_embed with shape (seq_len, batch_size, hidden_dim)
            encoder_pos_embed = get_sinusoidal_positional_embedding(
                batch_size, transformed_seq_len, hidden_dim, memory.device
            ).permute(1, 0, 2)
        
        query_pos = self.query_pos_embed_weight.unsqueeze(1).repeat(1, batch_size, 1)
        
        # If no initial query embeddings provided, use zeros
        if initial_query_embeddings is None:
            tgt = torch.zeros_like(query_pos)
        else:
            tgt = initial_query_embeddings.permute(1, 0, 2)
        
        # PyTorch Hub DETR decoder forward signature:
        # def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
        #             memory_key_padding_mask=None, pos=None, query_pos=None)
        decoder_outputs = self.decoder(
            tgt=tgt,                        # Query content
            memory=memory,                  # Encoder outputs
            pos=encoder_pos_embed,          # Position embeddings for encoder outputs
            query_pos=query_pos             # Position embeddings for queries
        )
        
        
        last_hidden_state = decoder_outputs[-1]  # Get last layer, shape: [num_queries, batch_size, hidden_dim] 
        last_hidden_state = last_hidden_state.permute(1, 0, 2)  # -> [batch_size, num_queries, hidden_dim]
        
        # Apply classification and bbox heads
        raw_logits = self.classification_head(last_hidden_state)
        raw_pred_boxes = self.bbox_regression_head(last_hidden_state).sigmoid()
        
        # If bbox_regression_head outputs a list of tensors (one per layer), get the last one
        if isinstance(raw_pred_boxes, list):
            raw_pred_boxes = raw_pred_boxes[-1]
            
        return raw_logits, raw_pred_boxes

    def predict_with_encodings(
        self,
        input: torch.Tensor,
        original_image_size: Tuple[int, int],
        query_embeds: torch.Tensor = None,
        encoder_pos_embed: torch.Tensor = None,
        threshold: float = 0.9,
        ):
        # Set model to evaluation mode
        self.eval()
        with torch.no_grad():
            raw_logits, raw_pred_boxes = self.forward(
                input=input,
                encoder_pos_embed=encoder_pos_embed,
                initial_query_embeddings=query_embeds,
            )

        # Get probabilities from logits
        prob = F.softmax(raw_logits, dim=-1)
        scores, labels = prob[..., :-1].max(-1) # Remove "no object" class

        # Keep only predictions above threshold
        keep = scores > threshold
        
        # Process each item in the batch
        final_detections = []
        
        for i in range(raw_logits.shape[0]): # For each item in batch
            # Get boxes, scores and labels for this item that are above threshold
            boxes_item = raw_pred_boxes[i, keep[i]]
            scores_item = scores[i, keep[i]]
            labels_item = labels[i, keep[i]]
            
            # Convert normalized boxes to pixel coordinates
            h, w = original_image_size
            boxes_item = box_cxcywh_to_xyxy(boxes_item)
            boxes_item = boxes_item * torch.tensor([w, h, w, h], device=boxes_item.device)
            
            # Create detection objects
            img_detections = []
            for score, label_id, box in zip(scores_item, labels_item, boxes_item):
                box_list = [round(float(i), 2) for i in box.tolist()]
                try:
                    # For PyTorch Hub model, we need a custom mapping
                    # You may need to add your own label mapping here
                    label = f"class_{label_id.item()}"
                except KeyError:
                    label = f"Unknown_Label_ID_{label_id.item()}"

                img_detections.append(
                    {"score": round(score.item(), 3), "label": label, "box": box_list}
                )
            
            final_detections.append(img_detections)

        return final_detections
