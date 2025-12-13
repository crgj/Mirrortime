import torch
import torch.nn as nn
import tinycudann as tcnn

class TemporalNetwork(nn.Module):
    def __init__(self):
        super(TemporalNetwork, self).__init__()
        
        # 4D Hash Grid Encoding
        # Inputs: 4 dims (x, y, z, t)
        # Config chosen for balance of speed and quality
        self.encoding = tcnn.Encoding(
            n_input_dims=4,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.5,
            },
        )
        
        # Small MLP Decoder
        self.network = tcnn.Network(
            n_input_dims=32, # 16 levels * 2 features
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            },
        )

    def forward(self, xyz, shift_t):
        # xyz: [N, 3] normalized to [0, 1]
        # shift_t: [N, 1] normalized to [0, 1]
        
        # Concatenate spatial and temporal inputs
        # t is repeated to match N if scalar, or passed as [N, 1]
        if shift_t.shape[0] != xyz.shape[0]:
             shift_t = shift_t.repeat(xyz.shape[0], 1)
             
        in_coords = torch.cat([xyz, shift_t], dim=-1)
        
        # Encoding
        enc = self.encoding(in_coords)
        
        # MLP
        out = self.network(enc)
        
        return out
