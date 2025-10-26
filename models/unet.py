import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size = 3,
            padding = 1
        )

        self.norm = nn.GroupNorm(
            32,
            out_channels
        )

    def forward (self, x):
        x = self.conv(x)
        x = self.norm(x)

        return F.gelu(x)

class DoubleConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()

        self.convblock1 = ConvBlock (in_channels,out_channels)
        self.convblock2 = ConvBlock(out_channels,out_channels)
        self.time_emb_proj = nn.Linear(time_dim, out_channels)
    
    def forward(self, x, emb):
        x = self.convblock1(x)

        t = self.time_emb_proj(F.gelu(emb))
        t = t.unsqueeze(-1).unsqueeze(-1)
        x = x + t

        x = self.convblock2(x)

        return x
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_dim):
        super().__init__()

        self.convblock1 = ConvBlock(in_channels, out_channels)
        self.convblock2 = ConvBlock(out_channels, out_channels)

        self.time_emb_proj = nn.Linear(t_dim, out_channels)

        if in_channels == out_channels:
            self.residual_conv = nn.Identity()
        else:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1)
        
    def forward(self, x, emb):
        
        h = self.convblock1(x)

        t = self.time_emb_proj(F.gelu(emb))
        t = t.unsqueeze(-1).unsqueeze(-1)

        h = h + t

        h = self.convblock2(h)

        return h + self.residual_conv(x)

class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(32, in_channels)

        self.qkv = nn.Conv2d(in_channels, in_channels*3,kernel_size=1)
        
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    
    def forward(self, x):
        b, c, h, w = x.shape
        
        h_norm = self.norm(x)

        qkv = self.qkv(h_norm)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        q = q.reshape(b, c, h * w).permute(0, 2, 1)
        k = k.reshape(b, c, h * w).permute(0, 2, 1)
        v = v.reshape(b, c, h * w).permute(0, 2, 1)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        attn_out = attn_out.permute(0, 2, 1).reshape(b, c, h, w)

        return x + self.proj_out(attn_out)

class Encoder(nn.Module):
    
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()

        #self.conv_block = DoubleConvBlock(in_channels, out_channels, time_dim)
        self.conv_block = ResBlock(in_channels, out_channels, time_dim)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    
    def forward(self, x, time_emb):
        skip = self.conv_block(x, time_emb)
        down = self.pool(skip)

        return skip, down
    
class Decoder(nn.Module):

    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        #self.conv_block = DoubleConvBlock(out_channels+out_channels, out_channels, time_dim)
        self.conv_block = ResBlock(out_channels+out_channels, out_channels, time_dim)

    def forward(self, x, skip_connection, time_emb):
        x = self.up(x)
        x = torch.cat([x, skip_connection], dim=1)

        return self.conv_block(x, time_emb)

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        if dim % 2 !=0:
            raise ValueError(f"Embedding dimension dim ({dim}) must be even.")
        self.dim = dim
    
    def forward(self, t):
        device = t.device

        half_dim = self.dim//2

        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * - emb)

        emb = t.unsqueeze(1)*emb.unsqueeze(0)
        emb = torch.cat([emb.sin(),emb.cos()],dim=1)

        return emb


class UNet(nn.Module):

    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()

        self.time_dim = time_dim

        self.time_embedding = SinusoidalTimeEmbedding(time_dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim*4),
            nn.GELU(),
            nn.Linear(time_dim*4, time_dim)
        )

        self.Encoder1 = Encoder(in_channels, 64, time_dim)
        self.Encoder2 = Encoder(64, 128, time_dim)
        self.Encoder3 = Encoder(128, 256, time_dim)
        self.Encoder4 = Encoder(256, 512, time_dim)

        self.bottleneck = ResBlock(512,1024, time_dim)
        self.bottleneck_attn = SelfAttentionBlock(1024)

        self.Decoder1 = Decoder(1024,512, time_dim)
        self.Decoder2= Decoder(512,256, time_dim)
        self.Decoder3 = Decoder(256,128, time_dim)
        self.Decoder4 = Decoder(128,64, time_dim)

        self.output = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x, t):

        t_emb = self.time_embedding(t)
        t_emb = self.time_mlp(t_emb.to(x.device))

        s1, d1 = self.Encoder1(x, t_emb)
        s2, d2 = self.Encoder2(d1, t_emb)
        s3, d3 = self.Encoder3(d2, t_emb)
        s4, d4 = self.Encoder4(d3, t_emb)

        b = self.bottleneck(d4, t_emb)
        b = self.bottleneck_attn(b)

        u4 = self.Decoder1(b,s4, t_emb)
        u3 = self.Decoder2(u4,s3, t_emb)
        u2 = self.Decoder3(u3,s2, t_emb)
        u1 = self.Decoder4(u2,s1, t_emb)

        out = self.output(u1)

        return out