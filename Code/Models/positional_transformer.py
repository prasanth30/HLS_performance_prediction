from BaseTransformer import BaseTransformer
from embedding_handler import EmbeddingHandler

class PositionalTransformer(BaseTransformer):
    def __init__(self, config):
        super(PositionalTransformer, self).__init__(
            embed_size=config['embed_size'],
            hidden_size=config['hidden_size'],
            n_heads=config['n_heads'],
            num_layers=config['num_layers'],
            n_outputs=config['n_outputs'],
            device=config['device']
        )
        self.embedding_handler = EmbeddingHandler(
            embed_size=config['embed_size'],
            key_embed_size=config['key_embed_size'],
            val_embed_size=config['val_embed_size'],
            num_encoding=config['num_encoding'],
            device=config['device'],
            pragmas=config['pragmas']
        )
        
    def get_embeddings(self, X_batch):
        embeddings_batch = []
        for X in X_batch:
            emb = self.embedding_handler.get_embeddings(X)
            if emb is not None:
                embeddings_batch.append(emb)
        return embeddings_batch
