


def get_model(name:str,exp_num:int):
    if name=='Set_Transformer':
        return SetTransformer(exp_num)
    elif name=='Periodic':
        return Periodic(exp_num)
    else:
        raise ValueError(f"Model {name} not found")