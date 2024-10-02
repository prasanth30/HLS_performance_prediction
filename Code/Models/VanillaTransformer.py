import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass
from typing import Any, Callable, Literal, cast


import math
import sys
sys.path.append('./../')
from utils import pragmas, target
class vanilla_transformer_model(nn.Module):
    def __init__(
        self,
        hidden_size=512,
        n_heads=8,
        n_outputs=7,
        key_embed_size=512,
        val_embed_size=256,
        embed_size=768,
        num_layers=4,  # Number of Transformer encoder layers
        device='cpu'
    ):
        super(vanilla_transformer_model, self).__init__()

        self.device = device
        self.embed_size = embed_size
        self.key_embed_size = key_embed_size
        self.val_embed_size = val_embed_size
        
        # Transformer Encoder Layer (Vanilla Transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size, 
            nhead=n_heads,
            dim_feedforward=hidden_size, 
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embed_size, n_outputs)
        )
        
        # Embeddings
        cat_table = {f'{label}_{value}': self.kaiming_init_embedding(embed_size) 
                     for label in pragmas.categorical_space for value in ['','off','flatten']}
        
        reg_table = {label: self.kaiming_init_embedding(key_embed_size) 
                     for label in pragmas.integer_space}
        
        self.reg_value_encoder = nn.ModuleDict({label: nn.Linear(1, val_embed_size) 
                                                for label in pragmas.integer_space})
        
        self.embeddings = nn.ParameterDict(cat_table | reg_table)
        
        for param in self.parameters():
            param.requires_grad = True

    def kaiming_init_embedding(self, size):
        embedding = torch.empty((size, 1), requires_grad=True, device=self.device)
        nn.init.kaiming_uniform_(embedding, a=math.sqrt(5))  # or use kaiming_normal_
        return embedding

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
            # Pre-allocate the embeddings tensor based on the number of embeddings and the embedding size
            embeddings = []
            for key, item in X.items():
                if item == '' or item is None:
                    continue
                if item == 'NA':
                    item = ''
                
                # Handle categorical embeddings
                if key in pragmas.categorical_space and item in ['off', 'flatten', '']:
                    embeddings.append(self.embeddings[f'{key}_{item}'].squeeze())
                
                # Handle integer embeddings
                elif key in pragmas.integer_space:
                    try:
                        scaled_value = (int(item)-1)/pragmas.mx_range[key]
                        value_embedding = self.reg_value_encoder[key](
                            torch.tensor(int(scaled_value), dtype=torch.float32, device=self.device).reshape(1, 1)
                        )
                        full_embedding = torch.cat([self.embeddings[key].squeeze(), value_embedding.squeeze()], dim=0)
                        embeddings.append(full_embedding)  # Add to the list of embeddings
                    except Exception as ex:
                        print(ex, key, item)
                else:
                    raise ValueError(f"Unknown Target {key} value: {item}")
            
            # Stack embeddings and add to batch
            if len(embeddings) > 0:
                embeddings_batch.append(torch.stack(embeddings))
        
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

        

if __name__ == "__main__":
    trans = vanilla_transformer_model()
    ret = trans([{'__PARA__L0':20, 
                '__PARA__L0_0':10},{'__PARA__L0_1':10, '__PARA__L0_1_0':100, 
                '__PARA__L0_2':1, '__PARA__L0_2_0':2}])
    options = PeriodicOptions()
    pc = Periodic(n_features=1,)