import torch
import torch.nn as nn 
import matplotlib.pyplot as plt
from module import Up, Down, SelfAttention, DoubleConv
import torch.optim as optim
import torchvision
import wandb
import numpy as np
from tqdm.auto import tqdm
import wandb 
from model_zoo import Unet
from diffusion_process import Diffusion
from nsynth_dataload import Nsynth, get_dataloader
import random 
import torchaudio
import sys 
import subprocess


if __name__== "__main__":
  
  device = 'cuda'
  sys.path.append('/home/daewoong/userdata/Study/diffusion/hifi_gan/diffwave/src/')
  from diffwave.inference import predict as diffwave_predict
 
  dataset = Nsynth()
  inst_dict = dataset.inst_dict
  pitch_dict = dataset.pitch_dict
  vel_dict = dataset.velocity_dict
  inst_specific = dataset.inst_specific
  
  wandb.init(project='Nsynth Diffusion_2')
  wandb.run.name = 'First train'
  wandb.run.save()
  
  args = {
    "learning_rate" : 1e-4,
    "epochs" : 5,
    "batch_size" : 32,
    "num_workers" : 3,
    # "scheduler" : "CosineAnnealingLR",
    # "T_max" : 1,
  }

  wandb.config.update(args)

  train_loader, valid_loader = get_dataloader(args['batch_size'], args['num_workers'])
  
  model = Unet()
  model = model.to(device)
  loss_fn = nn.MSELoss()
  optimizer = optim.AdamW(model.parameters(), lr=args['learning_rate'])
  #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['T_max'], eta_min= 1e-4)
  diffusion = Diffusion()
  a,b,c = torch.tensor([0]).to('cuda'), torch.tensor([43]).to('cuda'), torch.tensor([4]).to('cuda')
  labels = [a,b,c]
  #lr_plot = []

  for epoch in tqdm(range(args['epochs'])):
    
    model.train()
    train_step = 0
    for data in tqdm(train_loader):
      train_step += 1
      spectrogram, instrument, pitch, velocity = data
      
      spectrogram = spectrogram.to(device)
      instrument = instrument.to(device)
      pitch = pitch.to(device)
      velocity = velocity.to(device)
      y = [instrument, pitch, velocity]

      t = diffusion.sample_timesteps(spectrogram.shape[0]).to(device)
      diff_spec, noise = diffusion.noise_images(spectrogram, t)
      
      if np.random.random() < 0.1:
        y = None 
        
      predicted_noise = model(diff_spec, t, y = y)
      loss = loss_fn(predicted_noise, noise)
      wandb.log({"Training loss": loss.item()})
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    
    #lr_plot.append(optimizer.param_groups[0]["lr"]) 
    #scheduler.step()
    
    model.eval()
    val_loss = 0
    val_set_len = 0
    
    with torch.no_grad():
      for data in tqdm(valid_loader):
        spectrogram, instrument, pitch, velocity = data
              
        spectrogram = spectrogram.to(device)
        instrument = instrument.to(device)
        pitch = pitch.to(device)
        velocity = velocity.to(device)
        y = [instrument, pitch, velocity]
        
        val_set_len += spectrogram.shape[0]
        
        t = diffusion.sample_timesteps(spectrogram.shape[0]).to(device)
        diff_spec, noise = diffusion.noise_images(spectrogram, t)
        if np.random.random() < 0.1:
          y = None 
        predicted_noise = model(diff_spec, t, y = y)
        loss = loss_fn(predicted_noise, noise)
        val_loss += loss.item() * spectrogram.shape[0]
        
    wandb.log({"Validation loss": val_loss / val_set_len})
    

    torch.save(model.state_dict(), f"/home/daewoong/userdata/Study/diffusion/nsynth_diffusion/save_pt_2/model_{epoch+1}.pth") 
    sample = diffusion.sampling(model = model, batch_n =1, labels=labels)
    plt.imshow(sample[0][0].cpu().numpy(), aspect='auto', origin='lower')
    save_path = f"/home/daewoong/userdata/Study/diffusion/nsynth_diffusion/save_pt_2/sample_{epoch+1}.png"
    plt.savefig(save_path)
    wandb.log({"Sample": save_path})
  #subprocess.run(['python', 'inference.py', '--epochs', str(epoch+1)])    
  #wandb.log({"Learning rate": lr_plot})

