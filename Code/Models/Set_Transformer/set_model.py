import torch
import torch.nn as nn
import math
import sys

sys.path.append('./Models/Set_Transformer/set_transformer/')
from modules import MAB, SAB, ISAB, PMA
from utils import pragmas
class set_transformer_model(nn.Module):
    def __init__(
        self,
        dim_input=768,
        hidden_size=512,
        n_heads=8,
        num_outputs=7,
        n_class=1,
        ln=False,
        key_embed_size=512,
        val_embed_size=256,
        embed_size=768,
        num_inds=32,
        device='cpu'
    ):
        super(set_transformer_model, self).__init__()

        # Encoder
        self.enc = nn.Sequential(
            SAB(dim_input, hidden_size, n_heads, ln=ln),
            SAB(hidden_size, hidden_size, n_heads, ln=ln),
            SAB(hidden_size, hidden_size, n_heads, ln=ln),
            SAB(hidden_size, hidden_size, n_heads, ln=ln),
        )

        # Decoder
        self.dec = nn.Sequential(
            PMA(hidden_size, n_heads, num_outputs, ln=ln),
            nn.Linear(hidden_size, n_class)
        )

        # Move encoder and decoder to device
        self.enc = self.enc.to(device=device)
        self.dec = self.dec.to(device=device)
        
        # Embeddings
        self.device = device
        self.embed_size = embed_size
        self.key_embed_size = key_embed_size
        self.val_embed_size = val_embed_size
        
        # Example of how pragmas might look
        # pragmas = {'categorical_space': ['cat1', 'cat2'], 'integer_space': ['int1', 'int2']}
        
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

    def forward(self, X):
        X = X[0]
        # Pre-allocate the embeddings tensor based on the number of embeddings and the embedding size
        embeddings = torch.zeros((len(X), self.embed_size), device=self.device)
        
        idx = 0  # To track the position in the tensor
        for key, item in X.items():
            if item == '' or item is None:
                continue
            if item == 'NA':
                item = ''
            
            # Handle categorical embeddings
            if key in pragmas.categorical_space and item in ['off', 'flatten', '']:
                embeddings[idx] = self.embeddings[f'{key}_{item}'].squeeze()
            
            # Handle integer embeddings
            elif key in pragmas.integer_space:
                try:
                    value_embedding = self.reg_value_encoder[key](
                        torch.tensor(int(item), dtype=torch.float32, device=self.device).reshape(1, 1)
                    )
                    full_embedding = torch.cat([self.embeddings[key].squeeze(), value_embedding.squeeze()], dim=0)
                    embeddings[idx, :len(full_embedding)] = full_embedding  # Update only the relevant portion of the tensor
                except Exception as ex:
                    print(ex, key, item)
            
            else:
                raise ValueError(f"Unknown Target {key} value: {item}")
            
            idx += 1  # Move to the next index in the tensor

        # Reshape the tensor for the encoder
        embeddings = embeddings.unsqueeze(0)  # Adding batch dimension
        enc_out = self.enc(embeddings)
        dec_out = self.dec(enc_out)

        pred = torch.squeeze(dec_out, 0)
        pred = pred.reshape(-1)
        return pred
