import numpy as np
import torch.nn as nn

def orthogonal_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain=np.sqrt(2))
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)
    elif isinstance(m, nn.GRUCell):
        nn.init.orthogonal_(m.weight_ih.data)
        nn.init.orthogonal_(m.weight_hh.data)
        nn.init.zeros_(m.bias_ih.data)
        nn.init.zeros_(m.bias_hh.data)

class Dense(nn.Module):
    def __init__(self, in_dim, out_dim, act=nn.ELU):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.linear = nn.Linear(in_dim, out_dim)
        self.act = act() if act else nn.Identity()

    def forward(self, x):
        x = self.ln(x)
        x = self.linear(x)
        x = self.act(x)
        return x