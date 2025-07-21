import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from pytorch_lightning import LightningModule
from VQVAE_H import VQVAE_H

class ResConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self, x):
        return F.relu(self.conv(x) + self.shortcut(x))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_dim, heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embedding_dim, heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, 4*embedding_dim),
            nn.ReLU(),
            nn.Linear(4*embedding_dim, embedding_dim)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src):
        src2 = self.norm1(src)
        src = src + self.dropout(self.attention(src2, src2, src2)[0])
        sec2 = self.norm2(src)
        src = src + self.dropout(self.ff(src2))
        return src

class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, depth, heads, dropout):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(TransformerEncoderLayer(embedding_dim, heads, dropout))

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        return src

class Encoder(nn.Module):
    def __init__(self, input_channels, hidden_dims, embedding_dim, num_res_blocks, transformer_depth, transformer_heads):
        super().__init__()
        self.res_blocks = nn.Sequential(
            *[ResConv(input_channels if i==0 else hidden_dims, hidden_dims) for i in range(num_res_blocks)]
        )
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(hidden_dims, embedding_dim, kernel_size=1),
            Rearrange('b c h w -> b (h w) c', h=32, w=64)
        )
        self.positional_encoding = nn.Parameter(torch.randn(1, 32*64, embedding_dim))
        self.transformer = TransformerEncoder(embedding_dim, transformer_depth, transformer_heads, dropout=0.1)
        self.unpatch_embedding = Rearrange('b (h w) c -> b c h w', h=32, w=64)

    def forward(self, x):
        x = self.res_blocks(x)
        x = self.to_patch_embedding(x)
        x += self.positional_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        x = self.unpatch_embedding(x)
        return x

class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=None):
        super().__init__()
        self.upsample = upsample
        self.conv = ResConv(in_channels, out_channels)
    
    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, mode='nearest', scale_factor=self.upsample)
        x = self.conv(x)
        return x

class Decoder(nn.Module):
    def __init__(self, embedding_dim, output_channels, hidden_dims, transformer_depth, transformer_heads, num_res_blocks):
        super().__init__()
        self.patch_embedding = Rearrange('b c h w -> b (h w) c', h=32, w=64)
        self.transformer = TransformerEncoder(embedding_dim, transformer_depth, transformer_heads, dropout=0.1)
        self.to_conv_embedding = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=32, w=64),
            nn.Conv2d(embedding_dim, hidden_dims, kernel_size=1)
        )
        self.upsample_blocks = nn.Sequential(
            *[UpsampleConvLayer(hidden_dims if i==0 else hidden_dims, hidden_dims, upsample=None) for i in range(num_res_blocks)]
        )
        self.final_conv = nn.Conv2d(hidden_dims, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer(x)
        x = self.to_conv_embedding(x)
        x = self.upsample_blocks(x)
        x = self.final_conv(x)
        return x

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        super().__init__()
        self.beta = beta # weight of Commitment loss
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, x):
        x_flat = x.permute(0, 2, 3, 1).contiguous()
        batch_size, H, W, _ = x_flat.shape
        x_flat = x_flat.view(-1, self.embedding_dim)
        distances = (torch.sum(x_flat**2, dim=1, keepdim=True)
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2* torch.matmul(x_flat, self.embedding.weight.t()))
        indices = torch.argmin(distances, dim=1).unsqueeze(1)
        indices = indices.view(batch_size, H, W)

        x_quantized = self.embedding(indices).permute(0, 3, 1, 2).contiguous()
        e_latent_loss = F.mse_loss(x_quantized.detach(), x)
        q_latent_loss = F.mse_loss(x_quantized, x.detach())
        loss = q_latent_loss + self.beta * e_latent_loss
        x_quantized = x + (x_quantized - x).detach()

        return x_quantized, loss, indices

def load_quantized_from_checkpoint(model, checkpoint_path):
    model_pretrained = model.load_from_checkpoint(checkpoint_path)
    quantizer = model_pretrained.quantizer
    return quantizer

class VQVAE_L(LightningModule):
    def __init__(self, input_channels=1, output_channels=1, hidden_dims=256, embedding_dim=512, num_embeddings=1024,
                       num_res_blocks=6, transformer_depth=4, transformer_heads=8, learning_rate=4e-5):
        super().__init__()
        self.lr = learning_rate
        self.encoder = Encoder(input_channels=input_channels, hidden_dims=hidden_dims, embedding_dim=embedding_dim, 
                               num_res_blocks=num_res_blocks, transformer_depth=transformer_depth, transformer_heads=transformer_heads)
        self.decoder = Decoder(embedding_dim=embedding_dim, hidden_dims=hidden_dims, output_channels=output_channels, 
                               num_res_blocks=num_res_blocks, transformer_depth=transformer_depth, transformer_heads=transformer_heads)
        self.quantizer = load_quantized_from_checkpoint(VQVAE_H, 'vq_vae_high.ckpt')
        for param in self.quantizer.parameters():
            param.requires_grad = False
        # self.quantizer = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.reconstruction_loss = nn.MSELoss()
    
    def forward(self, x):
        z = self.encoder(x)
        z_q, commitment_loss, _ = self.quantizer(z)
        x_recon = self.decoder(z_q)
        return x_recon, z, z_q, commitment_loss

    def training_step(self, batch, batch_idx):
        x, y, variables, out_variables = batch
        x_recon, z, z_q, commitment_loss = self.forward(x)
        recon_loss = self.reconstruction_loss(x_recon, x)
        loss = recon_loss + commitment_loss
        self.log("commitment_loss", commitment_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("recon_loss", recon_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, variables, out_variables = batch
        x_recon, z, z_q, commitment_loss = self.forward(x)
        recon_loss = self.reconstruction_loss(x_recon, x)
        loss = recon_loss + commitment_loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,self.parameters()), lr=self.lr)
        return optimizer