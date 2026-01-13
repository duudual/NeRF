import torch
import torch.nn as nn


class Embedder:
    """Positional encoding embedder."""
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0, input_dims=3):
    """Get positional encoding embedder for positions/directions.
    
    Args:
        multires: log2 of max freq for positional encoding
        i: set 0 for default positional encoding, -1 for none
        input_dims: input dimension (3 for position, 3 for direction)
    
    Returns:
        embed: embedding function
        out_dim: output dimension
    """
    if i == -1:
        return nn.Identity(), input_dims
    
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


def get_time_embedder(multires_time, include_input=True):
    """Get positional encoding embedder specifically for time.
    
    Args:
        multires_time: log2 of max freq for time positional encoding
        include_input: whether to include input in embedding
    
    Returns:
        embed: embedding function
        out_dim: output dimension
    """
    if multires_time <= 0:
        # No encoding, just return identity
        return lambda x: x, 1
    
    embed_kwargs = {
        'include_input': include_input,
        'input_dims': 1,
        'max_freq_log2': multires_time - 1,
        'num_freqs': multires_time,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim
