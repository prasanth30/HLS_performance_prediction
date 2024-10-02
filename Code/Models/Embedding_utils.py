import sys
import math
sys.path.append('/Code/')
from utils import pragmas
from torch import Tensor
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Any, Callable, Literal, cast
import traceback
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

class Periodic_Embedder(nn.Module):
    def __init__(self,n_features:int,options:PeriodicOptions):
        super().__init__()
        self.periodic=Periodic(n_features,options)
        
    def forward(self,x:Tensor)->Tensor:
        return self.periodic(x)
    
class KeyValEmbedder(nn.Module):
    def __init__(self, pragmas, key_embed_size, val_embed_size, device='cpu',trainable=True):
        super(KeyValEmbedder,self).__init__()
        self.device = device
        self.trainable = trainable
        self.embed_size = key_embed_size + val_embed_size
        
        self.pragmas = pragmas
        cat_table = {f'{label}_{value}': self.kaiming_init_embedding(self.embed_size) 
                     for label in self.pragmas.categorical_space for value in ['','off','flatten']}
        
        reg_table = {label: self.kaiming_init_embedding(key_embed_size) 
                     for label in self.pragmas.integer_space}
        
        self.reg_value_encoder = nn.ModuleDict({label: nn.Linear(1, val_embed_size).to(self.device) 
                                                for label in self.pragmas.integer_space})
        
        self.embeddings = nn.ParameterDict(cat_table | reg_table).to(self.device)
        
        if self.trainable:
            for param in self.embeddings.parameters():
                param.requires_grad = True
        
    def kaiming_init_embedding(self, size):
        embedding = torch.empty((size, 1), requires_grad=True, device=self.device)
        nn.init.kaiming_uniform_(embedding, a=math.sqrt(5))  # or use kaiming_normal_
        return embedding 
      
    def forward(self,x:dict)->Tensor:
        embeddings = []
        for key,item in x.items():
            if key in self.pragmas.categorical_space and item in ['off', 'flatten', '']:
                embeddings.append(self.embeddings[f'{key}_{item}'].squeeze())
            
            # Handle integer embeddings
            elif key in self.pragmas.integer_space:
                try:
                    scaled_value = (int(item)-1)/self.pragmas.mx_range[key]
                    value_embedding = self.reg_value_encoder[key](
                        torch.tensor(int(scaled_value), dtype=torch.float32, device=self.device).reshape(1, 1)
                    )
                    full_embedding = torch.cat([self.embeddings[key].squeeze(), value_embedding.squeeze()], dim=0).to(self.device)
                    embeddings.append(full_embedding)  # Add to the list of embeddings
                except Exception as ex:
                    print(ex, key, item)
            else:
                raise ValueError(f"Unknown Target {key} value: {item}")
        embeddings_tensor = torch.stack(embeddings).to(self.device)
        return embeddings_tensor
    
if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    emb = KeyValEmbedder(pragmas,10,10,device)
    print(emb.forward({'__PARA__L0_1':10, '__PARA__L0_1_0':100, 
                '__PARA__L0_2':1, '__PARA__L0_2_0':2}))