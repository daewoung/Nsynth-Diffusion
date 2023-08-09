import torch 
import torch.nn as nn 

class DoubleConv(nn.Module):
  def __init__(self, in_c, out_c, mid_c=None, residual=False):
    super().__init__()
    
    self.residual = residual
    
    if not mid_c:
      mid_c = out_c
      
    self.gelu = nn.GELU()
      
    self.double_conv = nn.Sequential(
      nn.Conv2d(in_c, mid_c, kernel_size=3, padding=1, bias=False),
      nn.GroupNorm(num_groups=1, num_channels=mid_c),
      nn.GELU(),
      nn.Conv2d(mid_c, out_c, kernel_size=3, padding=1, bias=False),
      nn.GroupNorm(1, out_c),
    )
    
  def forward(self, x):
    if self.residual:
      return self.gelu(x + self.double_conv(x))
    
    else:
      return self.double_conv(x)
    
class Down(nn.Module):
  def __init__(self, in_c, out_c, emb_dim=256):
    super().__init__()
    
    self.maxpool_conv = nn.Sequential(
      nn.MaxPool2d(2),
      DoubleConv(in_c, in_c, residual=True),
      DoubleConv(in_c, out_c)
    )
    
    self.emb_layer = nn.Sequential(
      nn.SiLU(),
      nn.Linear(emb_dim, out_c)
    )
    
  def forward(self, x, t):
    x = self.maxpool_conv(x)
    emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
    
    return x + emb 
  
class Up(nn.Module):
  def __init__(self, in_c, out_c, emb_dim=256, odd = None):
    super().__init__()
    
    if odd:
      self.up = nn.Upsample(size= odd, mode='bilinear', align_corners=True)
    else:
      self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    self.conv = nn.Sequential(
      DoubleConv(in_c, in_c, residual=True),
      DoubleConv(in_c, out_c, mid_c = in_c//2),
    )
    self.emb_layer = nn.Sequential(
      nn.SiLU(),
      nn.Linear(emb_dim, out_c)
    )
    
  def forward(self, x, skip_x, t):
    x = self.up(x)
    x = torch.cat([skip_x, x], dim=1)
    x = self.conv(x)
    emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
    return x + emb 
  


# class SelfAttention(nn.Module):
#   def __init__(self, channels, size):
#     super().__init__()

#     self.channels = channels
#     self.size = size
#     self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=4, batch_first=True)
#     self.ln = nn.LayerNorm(channels)
#     self.ff = nn.Sequential(
#       nn.LayerNorm(channels),
#       nn.Linear(channels, channels),
#       nn.GELU(),
#       nn.Linear(channels, channels)
#     )
    
#   def forward(self, x):
#     x = x.view(-1, self.channels, self.size*self.size).swapaxes(1, 2) # (B x H*W x C)
#     x_ln = self.ln(x)
#     attention_value, _ = self.mha(x_ln, x_ln, x_ln)
#     attention_value = attention_value + x
#     attention_value = self.ff(attention_value) + attention_value
#     return attention_value.swapaxes(2,1).view(-1, self.channels, self.size, self.size)


class SelfAttention(nn.Module):
  def __init__(self, channels, mel_size, time_size):
    super().__init__()

    self.channels = channels
    self.mel_size = mel_size
    self.time_size = time_size
    self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=4, batch_first=True)
    self.ln = nn.LayerNorm(channels)
    self.ff = nn.Sequential(
      nn.LayerNorm(channels),
      nn.Linear(channels, channels),
      nn.GELU(),
      nn.Linear(channels, channels)
    )
    
  def forward(self, x):
    x = x.view(-1, self.channels, self.mel_size*self.time_size).swapaxes(1, 2) # (B x H*W x C)
    x_ln = self.ln(x)
    attention_value, _ = self.mha(x_ln, x_ln, x_ln)
    attention_value = attention_value + x
    attention_value = self.ff(attention_value) + attention_value
    return attention_value.swapaxes(2,1).view(-1, self.channels, self.mel_size, self.time_size)
  