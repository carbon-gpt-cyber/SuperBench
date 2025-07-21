import torch
from pytorch_lightning import LightningModule, Trainer
# import matplotlib.pyplot as plt 
import os
import matplotlib.pyplot as plt  
import yaml
import torch.nn as nn
from einops.layers.torch import Rearrange
from torchvision.transforms import transforms
from VQVAE_H import VQVAE_H
from VQVAE_L import VQVAE_L
from f_loss import PhysiFit
from einops import rearrange, repeat

from STtransformer import Step_t
import numpy as np
from datamodule import ClimateDownscalingDataModule

def lat_weighted_rmse(pred, y, transform, lat, variables=["geopotential_500","temperature_850","2m_temperature","10m_u_component_of_wind","10m_v_component_of_wind"]):
    b, t, c, w, h = pred.shape
    pred = pred.view(b*t, c, w, h)
    y = y.view(b*t, c, w, h)
    pred = transform(pred)
    y = transform(y)
    error = (pred - y) ** 2

    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(variables):
            loss_dict[var] = torch.mean(torch.sqrt(torch.mean(error[:, i] * w_lat, dim=(-2, -1))))
    loss_dict["loss"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])
    return loss_dict

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, D, H, W)
        x = self.proj(x)  # Project patches to embed_dim
        x = rearrange(x, 'b e d h w -> b (d h w) e')  # Flatten the patches
        return x

class SpaceTimeAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.spatial_transformer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.temporal_transformer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
    
    def forward(self, x, spatial_pos_encoding, temporal_pos_encoding, batch_size, time, height, width, spatial=True):
        if spatial:
            x = x + spatial_pos_encoding
            x = x.flatten(2).permute(2, 0, 1) #(H*W, N, E)
            x = self.spatial_transformer(x)
            x = x.permute(1, 2, 0).view(batch_size, time, -1, height * width)
        else:
            x = x + temporal_pos_encoding
            x = rearrange(x, 'b t e n -> t (b n) e')
            x = self.temporal_transformer(x)
            x = rearrange(x, 't (b n) e -> (b t) e n', b=batch_size, n=height*width)
            x = x.view(batch_size*time, -1, height, width)
        return x

class ST_transformer(nn.Module):
    def __init__(self, input_channels=512, num_frames=6, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.num_frames = num_frames
        self.d_model = d_model
        self.spatial_pos_encoding = nn.Parameter(torch.randn(1, d_model, 1, 1))
        self.temporal_pos_encoding = nn.Parameter(torch.randn(1, num_frames, d_model))
        # self.input_conv = nn.Conv2d(input_channels, d_model, kernel_size=1, stride=1, padding=0)
        self.layers = nn.ModuleList([
            SpaceTimeAttentionLayer(d_model=d_model, nhead=nhead)
            for _ in range(num_layers)
        ])
        # self.output_conv = nn.Conv2d(d_model, input_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        batch_size, time, channel, height, width = x.shape
        x = x.view(batch_size * time, channel, height, width)
        # x = self.input_conv(x)

        for i, layer in enumerate(self.layers):
            is_spatial = i % 2 == 0
            x = layer(x, self.spatial_pos_encoding, self.temporal_pos_encoding[:, :time].unsqueeze(-1),
                      batch_size, time, height, width, spatial=is_spatial)
        
        # x = self.output_conv(x)
        x = x.view(batch_size, time, channel, height, width)
        return x



class Net(nn.Module):
    def __init__(
        self,
        input_size=(4, 32, 64),
        in_channels=512,
        patch_size=(2, 2, 2),
        hidden_size=768,
        depth=6,
        num_heads=12,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.input_size = input_size
        self.num_patches = np.prod([input_size[i] // patch_size[i] for i in range(3)])
        self.num_heads = num_heads
        self.depth = depth

        self.patch_to_embedding = PatchEmbed3D(patch_size, in_channels, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size*4,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, hidden_size))
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))

        # self.final_layer = nn.Linear(hidden_size, in_channels)
        self.unproj = nn.ConvTranspose3d(hidden_size, in_channels, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        b, c, t, h, w = x.shape
        x = self.patch_to_embedding(x)
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embed[:, :x.size(1)]
        x = rearrange(x, 'b n d -> n b d')
        x = self.transformer_encoder(x)
        x = rearrange(x[1:,:,:], '(t h w) b d -> b d t h w', t=self.input_size[0]//self.patch_size[0], h=self.input_size[1]//self.patch_size[1], w=self.input_size[2]//self.patch_size[2])
        x = self.unproj(x)
        x = x.permute(0, 2, 1, 3, 4)
        return x

class Step_t(LightningModule):
    def __init__(self, lat, mean, std):
        super().__init__()
        self.l_encoder = VQVAE_L().encoder
        self.l_quantizer, self.l_decoder = VQVAE_L().quantizer, VQVAE_L().decoder
        self.h_encoder = VQVAE_H().encoder
        self.h_quantizer, self.h_decoder = VQVAE_H().quantizer, VQVAE_H().decoder
        self.net = Net()
        self.f_loss = PhysiFit(net_type='f').transformer
        self.b_loss = PhysiFit(net_type='b').transformer
        self.set_test(mean, std, lat)
    
    def set_test(self, mean, std, lat):
        mean_denorm, std_denorm = -mean / std, 1 / std
        self.denormalization = transforms.Normalize(mean_denorm, std_denorm)
        self.lat = lat

    def forward(self, x, y):
        batch_size, time, channel, hight, width = y.shape
        l_latent, h_latent = self.get_latent(x, y)
        h_latent_hat = self.forward_1(l_latent)
        h_latent_hat_m = rearrange(h_latent_hat, 'b t c h w -> (b t) c h w')
        y_hat = self.h_decoder(h_latent_hat_m).view(batch_size, time, channel, hight, width)
        return y_hat

    def forward_1(self, l_latent):
        h_latent_hat = self.net(l_latent)
        return h_latent_hat
    
    def get_latent(self, x, y):
        with torch.no_grad():
            batch_size, time, channel, hight, width = x.shape
            _, _, _, H_hight, H_width = y.shape
            x = x.view(batch_size*time, channel, hight, width)
            y = y.view(batch_size*time, channel, H_hight, H_width)
            l_latent = self.l_encoder(x)
            l_latent,_,_ = self.l_quantizer(l_latent)
            h_latent = self.h_encoder(y)
            h_latent,_,_ = self.h_quantizer(h_latent)
        return l_latent.view(batch_size, time, 512, 32, 64), h_latent.view(batch_size, time, 512, 32, 64)
    
    def save_feature_maps(self, features, batch_idx=0, time_idx=0, save_dir='./feature_maps'):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        num_features = features.shape[2]
        fig, axes = plt.subplots(1, num_features, figsize=(num_features * 5, 5))

        if num_features == 1:
            axes = [axes]
        
        for i in range(num_features):
            feature_map = features[batch_idx, time_idx, i]
            ax = axes[i]
            cax = ax.imshow(feature_map, cmap='RdBu', interpolation='lanczos', vmin=-1, vmax=1)
            ax.set_title(f'Feature: {i}')
            ax.axis('off')

        
        # fig.colorbar(cax)
        plt.savefig(os.path.join(save_dir, f'feature_maps_batch_{batch_idx}_time_{time_idx}'))
        plt.close(fig)
    
    def training_step(self, batch, batch_idx):
        pass 
    
    def validation_step(self, batch, batch_idx):
        x, y, variables, out_variables = batch
        y_hat = self(x, y)
        rmse_loss = lat_weighted_rmse(y_hat, y, self.denormalization, self.lat)

        if batch_idx == 1:
            self.save_feature_maps(y.to('cpu').numpy(), save_dir='./feature_maps/val/image_y')
            self.save_feature_maps(y_hat.to('cpu').numpy(), save_dir='./feature_maps/val/image_y_hat')

        for key in rmse_loss:
            self.log(f"{key}", rmse_loss[key], prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('val_loss', rmse_loss['loss'])
    
    def test_step(self, batch, batch_idx):
        x, y, variables, out_variables = batch
        y_hat = self(x, y)
        rmse_loss = lat_weighted_rmse(y_hat, y, self.denormalization, self.lat)

        if batch_idx == 10:
            self.save_feature_maps(y.to('cpu').numpy(), save_dir='./feature_maps/test/image_y')
            self.save_feature_maps(y_hat.to('cpu').numpy(), save_dir='./feature_maps/test/image_y_hat')

        for key in rmse_loss:
            self.log(f"{key}", rmse_loss[key], prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('test_loss', rmse_loss['loss'])


config = load_config('config.yaml')
data_module = ClimateDownscalingDataModule(**config['data_module'])
normalization = data_module.output_transforms
lat, lon = data_module.get_lat_lon()
mean_norm, std_norm = normalization.mean, normalization.std
params = {}
params.update({'lat': lat,
                        'mean': mean_norm,
                        'std': std_norm})
# load model
model = Step_t.load_from_checkpoint('my_model/best_model.ckpt', **params)
model.eval()

trainer = Trainer(precision=16)
trainer.validate(model=model, datamodule=data_module)
trainer.test(model=model, datamodule=data_module)
