import torch
import torch.nn as nn 
from module import Down, Up, SelfAttention, DoubleConv

class Unet(nn.Module):
  def __init__(self, c_in = 1, c_out = 1, first_c_in = 64, num_heads = 8, time_dim=256, num_labels = [28, 5, 112]):
    super().__init__()
    self.device = 'cuda'
    self.time_dim = time_dim
    
    self.inst_label_num = num_labels[0]
    self.velocity_label_num = num_labels[1]
    self.pitch_label_num = num_labels[2]
    
    self.mel_size = 80
    self.time_size = 251
    
    # self.inc = DoubleConv(c_in, 32)
    # self.down1 = Down(32, 128)
    self.inc = DoubleConv(c_in, first_c_in)
    self.down1 = Down(first_c_in, 128)

    self.sa1 = SelfAttention(128, mel_size = self.mel_size // 2, time_size = self.time_size // 2, num_heads=num_heads)
    self.down2 = Down(128, 256)
    self.sa2 = SelfAttention(256, mel_size = self.mel_size // 4, time_size = self.time_size // 4, num_heads=num_heads)
    self.down3 = Down(256, 256)
    self.sa3 = SelfAttention(256, mel_size = self.mel_size // 8, time_size = self.time_size // 8, num_heads=num_heads)
    
    self.bot1 = DoubleConv(256, 512)
    self.bot2 = DoubleConv(512, 512)
    self.bot3 = DoubleConv(512, 256)
    
    self.up1 = Up(512, 128)
    self.sa4 = SelfAttention(128, mel_size = self.mel_size // 4, time_size = self.time_size // 4, num_heads=num_heads)
    # self.up2 = Up(in_c = 256, out_c = 32, odd = (40, 125))
    # self.sa5 = SelfAttention(32, mel_size = self.mel_size // 2, time_size = self.time_size // 2)
    # self.up3 = Up(64, 32, odd = (80, 251))
    # self.sa6 = SelfAttention(32, mel_size = self.mel_size, time_size = self.time_size)
    # self.outc = nn.Conv2d(32, c_out, kernel_size=1)
    self.up2 = Up(in_c = 256, out_c = first_c_in, odd = (40, 125))
    self.sa5 = SelfAttention(first_c_in, mel_size = self.mel_size // 2, time_size = self.time_size // 2, num_heads=num_heads)
    self.up3 = Up(first_c_in*2, first_c_in, odd = (80, 251))
    self.sa6 = SelfAttention(first_c_in, mel_size = self.mel_size, time_size = self.time_size, num_heads=num_heads)
    self.outc = nn.Conv2d(first_c_in, c_out, kernel_size=1)
    
    if num_labels is not None:
      self.inst_emb = nn.Embedding(self.inst_label_num, time_dim)
      self.velocity_emb = nn.Embedding(self.velocity_label_num, time_dim)
      self.pitch_emb = nn.Embedding(self.pitch_label_num, time_dim)
    
  def pos_encoding(self, t, channels):
    inv_freq = 1.0 / (
        10000
        ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
    )
    pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
    pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
    pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
    return pos_enc
    
  def forward(self, x, t, y = None):
    t = t.unsqueeze(-1).type(torch.float)
    t = self.pos_encoding(t, self.time_dim) # [batch, time_dim]

    if y is not None:
      t += self.inst_emb(y[0])
      t += self.pitch_emb(y[1])
      t += self.velocity_emb(y[2])

    x1 = self.inc(x) # [batch, 64, 80, 251]
    x2 = self.down1(x1, t) # [1, 128, 40, 125]
    x2 = self.sa1(x2) # [batch, 128, 40, 125]
    x3 = self.down2(x2, t) #batch, 256, 20, 62]
    x3 = self.sa2(x3)
    
    x4 = self.down3(x3, t)
    x4 = self.sa3(x4)   # [batch, 256, 10, 31]
    

    
    x4 = self.bot1(x4) # [batch, 512, 10, 31]
    x4 = self.bot2(x4) # [batch, 512, 10, 31])
    x4 = self.bot3(x4) #torch.Size([1, 256, 10, 31])
    
    x = self.up1(x4, x3, t) # [batch, 128, 20, 62]
    x = self.sa4(x) #[batch, 128, 20, 62]

    
    x = self.up2(x, x2, t)  #([batch, 256, 40, 125])
    x = self.sa5(x)

    x = self.up3(x, x1, t)  #([batch, 64, 80, 251])

    x = self.sa6(x)
    output = self.outc(x)  #([batch, 1, 80, 251])
    
    return output