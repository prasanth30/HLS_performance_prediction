def set_trans_config(exp_num):
    if exp_num<=3:
        return {'key_embed_size' : 512,
        'val_embed_size' : 256,
        'embed_size' : 768, # Scale By Value
        # Encoder
        'dim_input' : 768,
        'n_heads' : 8,
        'hidden_size' : 512,
        'num_inds' : 32,
        'num_outputs' : 7,
        'n_class' : 1,
        'ln':False
        }
    else:
        return {
            'key_embed_size' : 512,
            'val_embed_size' : 256,
            'embed_size' : 768, # Scale By Value
            # Encoder
            'dim_input' : 768,
            'n_heads' : 8,
            'hidden_size' : 512,
            'num_inds' : 32,
            'num_outputs' : 7,
            'n_class' : 1,
            'ln':True
        }

def van_trans_config(exp_num):
    if exp_num<11:
        return {
                # Encoder
                'n_heads' : 8,
                'hidden_size' : 512,
                'n_outputs' : 7,
                'num_layers': 4,
                'lr':1e-3
            }   
    elif exp_num<=12: 
        return {
                'key_embed_size' : 512,
                'val_embed_size' : 256,
                'embed_size' : 768, # Scale By Value
                # Encoder
                'n_heads' : 8,
                'hidden_size' : 512,
                'n_outputs' : 7,
                'num_layers': 6
            }
    else:
        return {
                'key_embed_size' : 512,
                'val_embed_size' : 256,
                'embed_size' : 768, # Scale By Value
                # Encoder
                'n_heads' : 8,
                'hidden_size' : 512,
                'n_outputs' : 7,
                'num_layers': 10
            }

def pos_trans_config(exp_num):
    def_config = {'key_embed_size' : 512,
        'val_embed_size' : 256,
        'embed_size' : 768, # Scale By Value
        # Encoder
        'n_heads' : 8,
        'hidden_size' : 512,
        'n_outputs' : 7,
        'num_layers': 6,
        'num_encoding': 'periodic',
        'lr':1e-5}
    
    if exp_num in [1, 3]:
        return def_config
    
    if exp_num==2:
        def_config['num_encoding'] = 'ReLU'
        return def_config
def get_config(model_name:str,exp_num:int=1):
    if model_name == 'Set_Transformer':
        return set_trans_config(exp_num)
    elif model_name == 'Vanilla_Transformer':
        return van_trans_config(exp_num)
    elif model_name == 'Positional_Transformer':
        return pos_trans_config(exp_num)