class model_CFG:
    # Embedding
    key_embed_size = 512
    val_embed_size = 256
    embed_size = 768 # Scale By Value
    # Encoder
    dim_input = 768
    n_heads = 8
    hidden_size = 512
    num_inds = 32
    num_outputs = 7
    n_class = 1
    ln=False