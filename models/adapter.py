import torch
import torch.nn as nn


class Adapter(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, seq_len: int):
        """
        Args:
            input_dim (int): The dimension of the input features from your custom encoder.
            output_dim (int): The expected hidden dimension of the DETR decoder (d_model).
            seq_len (int): The expected sequence length of the input features.
        """
        super().__init__()

        self.projection = nn.Linear(input_dim, output_dim)
        print(f"Initialized SimpleAdapter: Input Dim={input_dim}, Output Dim={output_dim}, Seq Len={seq_len}")

    def forward(self, encoder_output):
        """
        Args:
            encoder_output (torch.Tensor): The output tensor from your custom encoder.
                                           Expected shape: (batch_size, seq_len, input_dim)
                                           or (batch_size, input_dim, H, W) etc.

        Returns:
            torch.Tensor: The transformed features suitable for the DETR decoder.
                          Expected shape: (batch_size, seq_len, output_dim)
        """
        transformed_output = self.projection(encoder_output)

        return transformed_output