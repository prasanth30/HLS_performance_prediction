import torch
import torch.nn as nn
import math
from utils import pragmas
from Models.numerical_embeddings import Periodic, PeriodicOptions

class BaseTransformer(nn.Module):
    def __init__(self, embed_size, hidden_size, n_heads, num_layers, n_outputs, device):
        super(BaseTransformer, self).__init__()
        self.device = device
        self.embed_size = embed_size
        
        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size, nhead=n_heads, dim_feedforward=hidden_size, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Decoder
        self.decoder = nn.Sequential(nn.Linear(embed_size, n_outputs))

        self.embeddings = None  # to be initialized in the child class
        for param in self.parameters():
            param.requires_grad = True

    def pad_and_mask(self, batch, max_len):
        """Pad the batch and create a mask to ignore padded elements."""
        padded_batch = torch.zeros((len(batch), max_len, self.embed_size), device=self.device)
        mask = torch.ones((len(batch), max_len), dtype=torch.bool, device=self.device)  # 1 means include

        for i, embeddings in enumerate(batch):
            length = embeddings.size(0)
            padded_batch[i, :length] = embeddings  # Copy embeddings
            mask[i, :length] = False  # False means real tokens (not padded)
        
        return padded_batch, mask

    def forward(self, X_batch):
        embeddings_batch = self.get_embeddings(X_batch)
        max_len = max([emb.size(0) for emb in embeddings_batch])
        padded_embeddings, mask = self.pad_and_mask(embeddings_batch, max_len)

        enc_out = self.encoder(padded_embeddings, src_key_padding_mask=mask)
        dec_out = self.decoder(enc_out[:, 0, :])  # Taking the first token for classification

        return torch.squeeze(dec_out, 0)
    
    def get_embeddings(self, X_batch):
        """This method should be implemented in child classes"""
        raise NotImplementedError("Subclasses must implement this method")
