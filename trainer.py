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
from inference import min_max_normalize

if __name__== "__main__":
  
  device = 'cuda'
  sys.path.append('/home/daewoong/userdata/Study/diffusion/hifi_gan/diffwave/src/')
  # from diffwave.inference import predict as diffwave_predict
  # diff_model_dir = '/home/daewoong/userdata/Study/diffusion/hifi_gan/diffwave/src/diffwave/pretrain/weights-554567.pt'

  dataset = Nsynth()
  inst_dict = dataset.inst_dict
  pitch_dict = dataset.pitch_dict
  vel_dict = dataset.velocity_dict
  inst_specific = dataset.inst_specific
  
  wandb.init(project='Nsynth Diffusion_2')
  wandb.run.name = '9th_train_more_Deep'
  wandb.run.save()
  
  args = {
    "learning_rate" : 1e-4,
    "epochs" : 10,
    "batch_size" : 32,
    "num_workers" : 3,
    "loss" : "MSELoss",
    "optimizer" : "AdamW",
    "scheduler" : "None",
    "step_size" : "None",
    "gamma" : "None",
    "num_heads" : 8,
    "start_channels" : "32 -> 64"
    # "scheduler" : "CosineAnnealingLR",
    # "T_max" : 1,
  }

  wandb.config.update(args)

  train_loader, valid_loader = get_dataloader(args['batch_size'], args['num_workers'])
  seed = 42
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  
  model = Unet()
  model = model.to(device)
  loss_fn = nn.MSELoss()
  optimizer = optim.AdamW(model.parameters(), lr=args['learning_rate'])
  #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['T_max'], eta_min= 1e-4)
  #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args['step_size'], gamma=args['gamma'])
  diffusion = Diffusion()
  a,b,c = torch.tensor([0]).to('cuda'), torch.tensor([43]).to('cuda'), torch.tensor([4]).to('cuda')
  labels = [a,b,c]
  lr_plot = []
  train_step = 0
  
  for epoch in tqdm(range(args['epochs'])):
    model.train()
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
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      #scheduler.step()
      
      wandb.log({"Learning rate": optimizer.param_groups[0]["lr"],
                 "Training loss": loss.item(), "Step_is" : train_step})
                # , step = train_step)  
      # if train_step % args['step_size'] == 0:
      #   scheduler.step()

              
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
    torch.save({'model' : model.state_dict(), 'optimizer' : optimizer.state_dict()}, 
               f"/home/daewoong/userdata/Study/diffusion/nsynth_diffusion/save_pt_4/model_{epoch+1}.pth") 
    
    # if (epoch+1) % 2 == 0:
    #   # sample = diffusion.sampling(model = model, batch_n =1, labels=labels)
    #   # plt.imshow(sample[0][0].cpu().numpy(), aspect='auto', origin='lower')
    #   # save_path = f"/home/daewoong/userdata/Study/diffusion/nsynth_diffusion/save_pt_4/sample_{epoch+1}.png"
    #   # plt.savefig(save_path)
    #   # vocoder_result, _ = diffwave_predict(min_max_normalize(sample[0][0]), diff_model_dir)
    #   # torchaudio.save(f"/home/daewoong/userdata/Study/diffusion/nsynth_diffusion/save_pt_4/sample_{epoch+1}_vocoder.wav" , vocoder_result.to('cpu'), sample_rate=16000)
    #   # wandb.log({"Sample": wandb.Image(save_path),
    #   #            "Audio" : wandb.Audio(f"/home/daewoong/userdata/Study/diffusion/nsynth_diffusion/save_pt_4/sample_{epoch+1}_vocoder.wav")}) #step = train_step)
    
    # elif (epoch+1) == 1:
    #   sample = diffusion.sampling(model = model, batch_n =1, labels=labels)
    #   torch.save(sample, f"/home/daewoong/userdata/Study/diffusion/nsynth_diffusion/save_pt_4/sample_{epoch+1}.pt")
    #   plt.imshow(sample[0][0].cpu().numpy(), aspect='auto', origin='lower')
    #   save_path = f"/home/daewoong/userdata/Study/diffusion/nsynth_diffusion/save_pt_4/sample_{epoch+1}.png"
    #   plt.savefig(save_path)
    #   vocoder_result, _ = diffwave_predict(min_max_normalize(sample[0][0]), diff_model_dir)
    #   torchaudio.save(f"/home/daewoong/userdata/Study/diffusion/nsynth_diffusion/save_pt_4/sample_{epoch+1}_vocoder.wav" , vocoder_result.to('cpu'), sample_rate=16000)
    #   wandb.log({"Sample": wandb.Image(save_path),
    #              "Audio" : wandb.Audio(f"/home/daewoong/userdata/Study/diffusion/nsynth_diffusion/save_pt_4/sample_{epoch+1}_vocoder.wav")} )#, step = train_step)
      
    #wandb.alert(f'Epoch : {epoch+1}', f'val_loss : {val_loss / val_set_len}')
  #subprocess.run(['python', 'inference.py', '--epochs', str(epoch+1)])    
  #wandb.log({"Learning rate": lr_plot})

wandb.finish()