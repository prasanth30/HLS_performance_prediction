import torch
import torch.nn as nn
import math
from typing import Literal
from dataclasses import dataclass
from torch import Tensor
import sys
sys.path.append('/Code/')
from utils import pragmas

@dataclass
class PeriodicOptions:
    n: int # the output size is 2 * n
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



def sinusoidal_positional_embedding(token_sequence_size, token_embedding_dim, n=10000.0):

    if token_embedding_dim % 2 != 0:
        raise ValueError("Sinusoidal positional embedding cannot apply to odd token embedding dim (got dim={:d})".format(token_embedding_dim))

    T = token_sequence_size
    d = token_embedding_dim #d_model=head_num*d_k, not d_q, d_k, d_v

    positions = torch.arange(0, T).unsqueeze_(1)
    embeddings = torch.zeros(T, d)

    denominators = torch.pow(n, 2*torch.arange(0, d//2)/d) # 10000^(2i/d_model), i is the index of embedding
    embeddings[:, 0::2] = torch.sin(positions/denominators) # sin(pos/10000^(2i/d_model))
    embeddings[:, 1::2] = torch.cos(positions/denominators) # cos(pos/10000^(2i/d_model))

    return embeddings

class EmbeddingHandler:
    def __init__(self, embed_size, key_embed_size, val_embed_size, num_encoding, device):
        self.device = device
        self.embed_size = embed_size
        self.key_embed_size = key_embed_size
        self.val_embed_size = val_embed_size
        
        # Categorical Embeddings
        self.cat_table = {f'{label}_{value}': self.kaiming_init_embedding(embed_size) 
                          for label in pragmas.categorical_space for value in ['', 'off', 'flatten']}
        
        # Integer Embeddings
        self.reg_table = {label: self.kaiming_init_embedding(key_embed_size) 
                          for label in pragmas.integer_space}

        # Value Embeddings
        if num_encoding == 'periodic':
            po = PeriodicOptions(n=val_embed_size // 2, sigma=0.0, trainable=True, initialization='normal')
            self.reg_value_encoder = nn.ModuleDict({label: Periodic(1, po)
                                                    for label in pragmas.integer_space})
        elif num_encoding == 'Linear':
            self.reg_value_encoder = nn.ModuleDict({label: nn.Linear(1, val_embed_size)
                                                    for label in pragmas.integer_space})
        elif num_encoding == 'ReLU':
            relu1 = nn.ReLU()
            self.reg_value_encoder = nn.ModuleDict({label: nn.Sequential(
                nn.Linear(1, val_embed_size), relu1, nn.Linear(val_embed_size, val_embed_size))
                for label in pragmas.integer_space})
        
        self.embeddings = nn.ParameterDict(self.cat_table | self.reg_table)

    def kaiming_init_embedding(self, size):
        embedding = torch.empty((size, 1), requires_grad=True, device=self.device)
        nn.init.kaiming_uniform_(embedding, a=math.sqrt(5))  # or use kaiming_normal_
        return embedding

    def get_embeddings(self, X):
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
                scaled_value = (int(item) - 1) / pragmas.mx_range[key]
                value_embedding = self.reg_value_encoder[key](
                    torch.tensor(int(scaled_value), dtype=torch.float32, device=self.device).reshape(1, 1)
                )
                full_embedding = torch.cat([self.embeddings[key].squeeze(), value_embedding.squeeze()], dim=0)
                embeddings.append(full_embedding)

        return torch.stack(embeddings) if len(embeddings) > 0 else None
