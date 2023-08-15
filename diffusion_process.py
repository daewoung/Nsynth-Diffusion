import torch
import torch.nn as nn 
from tqdm.auto import tqdm
class Diffusion:
  def __init__(self, noise_steps=1000, beta_start = 1e-4, beta_end = 0.02, mel_size=80, time_size = 251, device = 'cuda'):
    self.noise_steps = noise_steps
    self.beta_start = beta_start
    self.beta_end = beta_end
    self.mel_size = mel_size
    self.time_size = time_size
    self.device = device

    self.beta = self.linear_noise_schedule().to(self.device)
    
    self.alpha = 1. - self.beta 
    self.alpha_hat = torch.cumprod(self.alpha, dim=0)
    
  def linear_noise_schedule(self):
    return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
  
  def noise_images(self, x, t):
    sqrt_alpha = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
    one_minus_sqrt_alpha = torch.sqrt(1. - self.alpha_hat[t])[:, None, None, None]
    epsilon = torch.randn_like(x).to(x.device)
    return sqrt_alpha * x + one_minus_sqrt_alpha * epsilon, epsilon

  def sample_timesteps(self, n):
    return torch.randint(low=1, high = self.noise_steps, size=(n,))

  def sampling(self, model, batch_n, labels, cfg_scale=3):
    
    model.eval()
    with torch.no_grad():
      x = torch.randn(batch_n, 1, self.mel_size, self.time_size).to(self.device)
      for t in tqdm(range(self.noise_steps -1 , 0, -1)): # 999, 998, ..., 1
        
        time = torch.tensor(t).repeat(batch_n).long().to(self.device)
        pred_noise = model(x, time, labels)
        
        if cfg_scale > 0:
          uncond_predicted_noise = model(x, time, None)
          pred_noise = torch.lerp(uncond_predicted_noise, pred_noise, cfg_scale)  
                  
        alpha = self.alpha[time][:, None, None, None]
        alpha_hat = self.alpha_hat[time][:, None, None, None]
        beta = self.beta[time][:, None, None, None]
        noise_z = torch.randn_like(x) if t > 1 else torch.zeros_like(x)
        x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * pred_noise) + torch.sqrt(beta) * noise_z
    return x                                                      
  