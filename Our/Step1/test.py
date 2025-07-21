from einops.layers.torch import Rearrange

import torch
a = torch.randn(1,512,128,256)
layer = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', h=32, w=64)
b = layer(a)
print(b.shape)