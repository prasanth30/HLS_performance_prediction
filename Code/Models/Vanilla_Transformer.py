import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass
from typing import Any, Callable, Literal, cast


import math
import sys
sys.path.append('/Code/')
from Models.Embedding_utils import KeyValEmbedder
from utils import pragmas, target

class Indvidual_Decoder(nn.Module): # 
    def __init__(self, embed_size, n_outputs,device):
        super(Indvidual_Decoder, self).__init__()
        self.device = device
        self.decoder_continuous = nn.ModuleDict({
            target: nn.Linear(embed_size, 1) for target in target.continuous_targets
        }).to(self.device)
        self.decoder_categorical = nn.ModuleDict({
            target: nn.Linear(embed_size, 2) for target in target.categorical_targets
        }).to(self.device)
        for param in self.parameters():
            param.requires_grad = True
    def forward(self, x):
        return torch.cat([self.decoder_categorical[target](x) for target in target.categorical_targets] + 
                         [self.decoder_continuous[target](x) for target in target.continuous_targets], dim=1)

class vanilla_transformer_model(nn.Module):
    def __init__(
        self,
        hidden_size=512,
        n_heads=8,
        n_outputs=7,
        num_layers=4,  # Number of Transformer encoder layers
        embedding_type='KeyVal',
        decoder_type='Indvidual',
        embedding_cfg={'key_embed_size':256, 'val_embed_size':256},
        device='cuda:1' if torch.cuda.is_available() else 'cpu'
    ):
        super(vanilla_transformer_model, self).__init__()

        self.device = device
        self.embed_size = embedding_cfg['key_embed_size']+embedding_cfg['val_embed_size']
        self.embedding_type = embedding_type
        self.embedding_cfg = embedding_cfg
        self.decoder_type = decoder_type
        # Transformer Encoder Layer (Vanilla Transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_size, 
            nhead=n_heads,
            dim_feedforward=hidden_size, 
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers).to(self.device)
        # Decoder
        if self.decoder_type == 'Linear':
            self.decoder = nn.Sequential(
                nn.Linear(self.embed_size, n_outputs)
            ).to(self.device)
        elif self.decoder_type == 'LReLu':
            self.decoder = nn.Sequential(
                nn.Linear(self.embed_size, self.embed_size),
                nn.ReLU(),
                nn.Linear(self.embed_size, n_outputs)
            ).to(self.device)
        elif self.decoder_type == 'Indvidual':
            self.decoder = Indvidual_Decoder(self.embed_size, n_outputs,self.device)

        self.embedder = KeyValEmbedder(pragmas,**self.embedding_cfg,device=self.device)

        
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
        # Batch input X_batch is a list of dictionaries with variable lengths of key-value pairs.
        embeddings_batch = []
        
        for X in X_batch:
            embeddings = self.embedder(X)
            embeddings_batch.append(embeddings)
        # Determine the maximum sequence length in the batch for padding
        max_len = max([emb.size(0) for emb in embeddings_batch])
        
        # Pad the sequences and create the mask
        padded_embeddings, mask = self.pad_and_mask(embeddings_batch, max_len)

        # Transformer encoding
        enc_out = self.encoder(padded_embeddings, src_key_padding_mask=mask)
        # Final decoding to prediction (take the first token's embedding for classification)
        dec_out = self.decoder(enc_out[:, 0, :])  # Taking the first token for classification
        
        pred = torch.squeeze(dec_out, 0)
        return pred

@dataclass
class PeriodicOptions:
    n: int  # the output size is 2 * n
    sigma: float
    trainable: bool
    initialization: Literal['log-linear', 'normal']

def cos_sin(x: Tensor) -> Tensor:
    return torch.cat([torch.cos(x), torch.sin(x)], -1)

class Periodic(nn.Module):
    def __init__(self, n_features: int, options: PeriodicOptions) -> None:
        super().__init__()
        if options.initialization == 'log-linear':
            coefficients = options.sigma ** (torch.arange(options.n) / options.n)
            coefficients = coefficients[None].repeat(n_features, 1)
        else:
            assert options.initialization == 'normal'
            coefficients = torch.normal(0.0, options.sigma, (n_features, options.n))
        if options.trainable:
            self.coefficients = nn.Parameter(coefficients)  # type: ignore[code]
        else:
            self.register_buffer('coefficients', coefficients)

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 2
        return cos_sin(2 * torch.pi * self.coefficients[None] * x[..., None])   


if __name__ == "__main__":
    trans = vanilla_transformer_model(embedding_cfg={'key_embed_size':4, 'val_embed_size':4},decoder_type='Indvidual')
    ret = trans([{'__PARA__L0':20, 
                '__PARA__L0_0':10},{'__PARA__L0_1':10, '__PARA__L0_1_0':100, 
                '__PARA__L0_2':1, '__PARA__L0_2_0':2}])
    print(ret)
    print(ret.shape)