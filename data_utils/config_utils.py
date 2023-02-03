
def parse_config(config):
    assert config['input_style'] in ["normed_raw", "log1p", "binned"]
    assert config['input_emb_style'] in ["category", "continuous", "scaling"]
    
    if config['input_style'] == "binned":
        if config['input_emb_style'] == "scaling":
            raise ValueError("input_emb_style `scaling` is not supported for binned input.")
    elif config['input_style']  in ["log1p","normed_raw"]:
        if config['input_emb_style'] == "category":
            raise ValueError(
                "input_emb_style `category` is not supported for log1p or normed_raw input."
            )
    
    if config['input_emb_style'] == "category":
        config['mask_value'] = config['n_bins'] + 1
        config['pad_value'] = config['n_bins']  # for padding gene expr values
        config['n_input_bins'] = config['n_bins'] + 2
    else:
        config['mask_value'] = -1
        config['pad_value'] = -2
        config['n_input_bins'] = config['n_bins']
    if config['ADV'] and config['DAB']:
        raise ValueError("ADV and DAB cannot be both True.")
    config['DAB_separate_optim'] = True if config['DAB'] > 1 else False