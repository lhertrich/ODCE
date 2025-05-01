import torch
import torch.nn as nn
from typing import Tuple, List, Dict, Any, Union
import math
import types
from transformers import DetrForObjectDetection, DetrImageProcessor
from adapter import Adapter


def get_sinusoidal_positional_embedding(batch_size, seq_len, hidden_dim, device):
    position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)  # (seq_len, 1)
    div_term = torch.exp(torch.arange(0, hidden_dim, 2, dtype=torch.float32, device=device) *
                         (-math.log(10000.0) / hidden_dim))  # (hidden_dim // 2)

    pe = torch.zeros(seq_len, hidden_dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    pe = pe.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, seq_len, hidden_dim)
    return pe


class AdapterDetr(nn.Module):

    def __init__(
        self,
        adapter: nn.Module,
        model_checkpoint: str = "facebook/detr-resnet-50",
        train_decoder: bool = True, # Option to train DETR decoder
        train_prediction_heads: bool = True, # Option to train prediction heads
        encoder: nn.Module = None
    ):

        super().__init__()
        self.adapter = adapter
        self.encoder = encoder

        print(f"Loading model and processor from {model_checkpoint}...")
        self.detr_full_model = DetrForObjectDetection.from_pretrained(model_checkpoint)
        self.config = self.detr_full_model.config # Contains num_labels, id2label etc.
        self.processor = DetrImageProcessor.from_pretrained(model_checkpoint)

        #Get decoder out of DETR model
        self.detr_model = self.detr_full_model.model
        self.decoder = self.detr_model.decoder

        # Initialize new prediction head
        num_user_labels   = 365
        num_model_classes = num_user_labels + 1
        hidden_dim = self.detr_model.config.d_model
        self.classification_head = nn.Linear(hidden_dim, num_model_classes)
        nn.init.xavier_uniform_(self.classification_head.weight)

        # FRESH LABEL MAPS IF NECESSARY: self.config.id2label, self.config.label2id

        # Keep bounding box head
        self.bbox_regression_head = self.detr_full_model.bbox_predictor

        # Learnable object query positional embeddings (part of DETR)
        # NB! don't understand
        self.query_pos_embed_weight = self.detr_model.query_position_embeddings.weight
        self.num_queries = self.query_pos_embed_weight.shape[0]
        self.hidden_dim = self.query_pos_embed_weight.shape[1]

        # classification and bbox heads out of DETR
        self.classification_head = self.detr_full_model.class_labels_classifier
        self.bbox_regression_head = self.detr_full_model.bbox_predictor

        # Freeze the DETR decoder and query embeddings
        if not train_decoder:
            for param in self.detr_model.parameters():
                param.requires_grad = False
            print("Froze DETR decoder parameters")
        else:
            print("DETR decoder trainable")

        # Freeze prediction heads if specified
        if not train_prediction_heads:
             for param in self.classification_head.parameters():
                  param.requires_grad = False
             for param in self.bbox_regression_head.parameters():
                  param.requires_grad = False
             print("DETR prediction heads frozen.")
        else:
             print("DETR prediction heads are trainable.")

        print("AdapterDetrModel initialized.")


    def forward(
        self,
        input: torch.Tensor,
        encoder_pos_embed: torch.Tensor = None, # Positional embeddings for encoder hidden states
        initial_query_embeddings: torch.Tensor = None # Initial input embeddings for queries
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        #If the input is given before encoding, encode it
        if self.encoder is not None:
            input = self.encoder(input)
        #Pass the encodings through the adapter
        transformed_encoder_hidden_states = self.adapter(input)

        batch_size, transformed_seq_len, hidden_dim = transformed_encoder_hidden_states.shape

        query_pos_embed = self.query_pos_embed_weight.unsqueeze(0).repeat(batch_size, 1, 1)
        if initial_query_embeddings is None:
            # Standard practice: Initialize decoder input embeddings to zeros
            initial_query_embeddings = torch.zeros_like(query_pos_embed)

        # Prepare encoder positional embeddings
        if encoder_pos_embed is None:
            encoder_pos_embed = get_sinusoidal_positional_embedding(batch_size, transformed_seq_len, hidden_dim, transformed_encoder_hidden_states.device)


        decoder_outputs = self.decoder(
            inputs_embeds=initial_query_embeddings,
            encoder_hidden_states=transformed_encoder_hidden_states,
            query_position_embeddings=query_pos_embed,
            position_embeddings=encoder_pos_embed,
        )

        last_hidden_state = decoder_outputs.last_hidden_state # Shape: (batch_size, num_queries, hidden_dim)

        raw_logits = self.classification_head(last_hidden_state) # Shape: (batch_size, num_queries, num_classes)
        raw_pred_boxes = self.bbox_regression_head(last_hidden_state) # Shape: (batch_size, num_queries, 4)

        # Return the raw outputs for loss calculation
        return raw_logits, raw_pred_boxes


    def predict_with_encodings(        
        self,
        input: torch.Tensor,
        original_image_size: Tuple[int, int],
        query_embeds: torch.Tensor = None, #Should be the query but what is it when no img
        encoder_pos_embed: torch.Tensor = None,
        threshold: float = 0.9
    ):
        self.detr_model.eval() # Set model to evaluation mode
        self.eval()
        with torch.no_grad():
            raw_logits, raw_pred_boxes = self.forward(
                 input=input,
                 encoder_pos_embed=encoder_pos_embed,
                 initial_query_embeddings=query_embeds
             )

        # Shape of the images must be passed as list
        target_sizes = torch.tensor(original_image_size, device=raw_logits.device) # Shape [batch_size, 2]

        #Formulate output object
        output_object = types.SimpleNamespace()
        output_object.logits = raw_logits
        output_object.pred_boxes = raw_pred_boxes

        # Use the loaded processor to generate a list, where each element
        # corresponds to an embedding in the batch.
        processed_results  = self.processor.post_process_object_detection(
            outputs=output_object,
            target_sizes=target_sizes,
            threshold=threshold
        )

        # Map label IDs to strings using the model's config
        final_detections = []
        for i, detections_dict in enumerate(processed_results):
            img_detections = []
            for score, label_id, box in zip(detections_dict['scores'], detections_dict['labels'], detections_dict['boxes']):
                box_list = [round(float(i), 2) for i in box]
                try:
                    label = self.detr_model.config.id2label[label_id.item()]
                except KeyError:
                    label = f"Unknown_Label_ID_{label_id.item()}"

                img_detections.append({
                    'score': round(score.item(), 3),
                    'label': label,
                    'box': box_list
                })
            final_detections.append(img_detections)

        return final_detections
        
        
        




