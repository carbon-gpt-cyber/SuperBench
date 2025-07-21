from pytorch_lightning import LightningModule
import torch.nn.functional as F
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.optim.lr_scheduler import _LRScheduler
from torchvision.transforms import transforms
import math

from VQVAE_H import VQVAE_H
from VQVAE_L import VQVAE_L
import numpy as np

def lat_weighted_mse(pred, y, lat, variables=["geopotential_500","temperature_850","2m_temperature","10m_u_component_of_wind","10m_v_component_of_wind"]):
    error = (pred - y) ** 2

    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(variables):
            loss_dict[var] = (error[:,i] * w_lat).mean()
    
    loss_dict["loss"] = (error * w_lat.unsqueeze(1)).mean(dim=1).mean()
    return loss_dict

def lat_weighted_rmse(pred, y, transform, lat, variables=["geopotential_500","temperature_850","2m_temperature","10m_u_component_of_wind","10m_v_component_of_wind"]):
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

class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * (self.last_epoch / self.warmup_epochs) for base_lr in self.base_lrs]
        else:
            cos_epoch = self.last_epoch - self.warmup_epochs
            cos_epochs = self.max_epochs - self.warmup_epochs
            lr = [base_lr * (1 + math.cos(math.pi * cos_epoch / cos_epochs)) / 2 for base_lr in self.base_lrs]
            lr = [max(l, 1e-7) for l in lr]
            return lr

class TransformerNet(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.rearrange_1 = Rearrange('b n c h w -> b (n h w) c')
        self.rearrange_2 = Rearrange('b c h w -> b (h w) c')
        layer_1 = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.layer_1 = nn.TransformerEncoder(layer_1, num_layers=num_layers//2)
        self.conv = nn.Conv2d(embed_dim*3, embed_dim, 1)
        layer_2 = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.layer_2 = nn.TransformerEncoder(layer_2, num_layers=num_layers//2)
    
    def forward(self, l_1_latent, l_2_latent, h_latent):
        b, c, h, w = h_latent.size()

        x = torch.stack([l_1_latent, l_2_latent, h_latent], dim=1) #(b, 3, c, h, w)
        x = self.rearrange_1(x) #(b, 3*h*w, c)

        x = self.layer_1(x.permute(1, 0, 2))
        x = x.permute(1, 0, 2)
        x = x.view(b, 3, h, w, c).permute(0, 1, 4, 2, 3).contiguous()
        x = x.view(b, 3*c, h, w)

        x = self.conv(x)     
        x = self.rearrange_2(x)
        x = self.layer_2(x.permute(1, 0, 2)) # h*w b c
        x = x.permute(1, 0, 2)  # b h*w c
        x = x.view(b, h, w, c).permute(0, 3, 1, 2)

        return x

def load_encoder_from_checkpoint(model, checkpoint_path):
    model_pretrained = model.load_from_checkpoint(checkpoint_path)
    return model_pretrained.encoder

def load_decoder_from_checkpoint(model, checkpoint_path):
    model_pretrained = model.load_from_checkpoint(checkpoint_path)
    quantizer = model_pretrained.quantizer
    decoder = model_pretrained.decoder
    return quantizer, decoder

class PhysiFit(LightningModule):
    def __init__(self, net_type, l_encoder_checkpoint_path, h_encoder_checkpoint_path, learning_rate, warmup_epochs, max_epochs, mean, std, lat):
        super().__init__()
        self.net_type = net_type
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.l_encoder = load_encoder_from_checkpoint(VQVAE_L(), l_encoder_checkpoint_path).eval()
        self.l_quantizer, _ = load_decoder_from_checkpoint(VQVAE_L(), l_encoder_checkpoint_path)
        self.l_quantizer = self.l_quantizer.eval()
        self.h_encoder = load_encoder_from_checkpoint(VQVAE_H(), h_encoder_checkpoint_path).eval()
        self.h_quantizer, self.h_decoder = load_decoder_from_checkpoint(VQVAE_H(), h_encoder_checkpoint_path)
        self.h_quantizer = self.h_quantizer.eval()
        self.h_decoder = self.h_decoder.eval()
        self.transformer = TransformerNet()
        self.freeze_param()
        self.set_test(mean, std, lat)
    
    def set_test(self, mean, std, lat):
        mean_denorm, std_denorm = -mean / std, 1 / std
        self.denormalization = transforms.Normalize(mean_denorm, std_denorm)
        self.lat = lat
    
    def freeze_param(self):
        for param in self.l_encoder.parameters():
            param.requires_grad = False
        for param in self.h_encoder.parameters():
            param.requires_grad = False
        for param in self.h_quantizer.parameters():
            param.requires_grad = False
        for param in self.h_decoder.parameters():
            param.requires_grad = False
        print("Freezed the param of encoder.")

    def forward(self, l_1_latent, l_2_latent, h_latent):
        out = self.transformer(l_1_latent, l_2_latent, h_latent)
        return out
    
    def get_latent(self, x, y):
        with torch.no_grad():
            l_1 = x[:,0,:,:,:]
            l_2 = x[:,1,:,:,:]
            h_1 = y[:,0,:,:,:]
            h_2 = y[:,1,:,:,:]
            l_1_latent = self.l_encoder(l_1)
            l_1_latent, _, _ = self.l_quantizer(l_1_latent)
            l_2_latent = self.l_encoder(l_2)
            l_2_latent, _, _ = self.l_quantizer(l_2_latent)
            h_1_latent = self.h_encoder(h_1)
            h_1_latent, _, _ = self.h_quantizer(h_1_latent)
            h_2_latent = self.h_encoder(h_2)
            h_2_latent, _, _ = self.h_quantizer(h_2_latent)
        return l_1_latent, l_2_latent, h_1_latent, h_2_latent


    def training_step(self, batch, batch_idx):
        x, y, variables, out_variables = batch
        l_1_latent, l_2_latent, h_1_latent, h_2_latent = self.get_latent(x, y)
        
        if self.net_type == 'f':
            h_latent_hat = self.forward(l_1_latent, l_2_latent, h_1_latent)
            y_hat = self.h_decoder(h_latent_hat)
            loss = torch.nn.functional.mse_loss(h_latent_hat, h_2_latent)
            # loss += 0.1*lat_weighted_mse(y_hat, y[:,1,:,:,:], self.lat)['loss']
        elif self.net_type == 'b':
            h_latent_hat = self.forward(l_1_latent, l_2_latent, h_2_latent)
            y_hat = self.h_decoder(h_latent_hat)
            loss = torch.nn.functional.mse_loss(h_latent_hat, h_1_latent)
            # loss += 0.1*lat_weighted_mse(y_hat, y[:,0,:,:,:], self.lat)['loss']
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, variables, out_variables = batch
        l_1_latent, l_2_latent, h_1_latent, h_2_latent = self.get_latent(x, y)
        if self.net_type == 'f':
            h_latent_hat = self.forward(l_1_latent, l_2_latent, h_1_latent)
            y_hat = self.h_decoder(h_latent_hat)
            mse_loss = lat_weighted_rmse(y_hat, y[:,1,:,:], self.denormalization, self.lat, variables=["2m_temperature"])
        elif self.net_type == 'b':
            h_latent_hat = self.forward(l_1_latent, l_2_latent, h_2_latent)
            y_hat = self.h_decoder(h_latent_hat)
            mse_loss = lat_weighted_rmse(y_hat, y[:,0,:,:,:], self.denormalization, self.lat, variables=["2m_temperature"])
        for key in mse_loss:
            self.log(f"{key}", mse_loss[key], prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('val_loss', mse_loss['loss']) 
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.transformer.parameters(), lr=self.learning_rate)
        # scheduler = {
        #     'scheduler': WarmupCosineAnnealingLR(optimizer, warmup_epochs=self.warmup_epochs, max_epochs=self.max_epochs),
        #     'name': 'learning_rate',
        #     'interval': 'epoch',
        #     'frequency': 1
        # }
        # return [optimizer], [scheduler]
        return optimizer
        