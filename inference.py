import torch 
import matplotlib.pyplot as plt
import random 
import torchaudio
import argparse
import sys
import numpy as np 
import wandb 
from model_zoo import Unet
from nsynth_dataload import Nsynth
from diffusion_process import Diffusion
from tqdm.auto import tqdm
from PIL import Image

def inference(diffusion, model, inst_dict, pitch_dict, vel_dict, inst_specific, epoch, batch_n = 28,  device = 'cuda'):
  model_dir = '/home/daewoong/userdata/Study/diffusion/hifi_gan/diffwave/src/diffwave/pretrain/weights-554567.pt'
  pth = '/home/daewoong/userdata/Study/diffusion/nsynth-train-all/audio/'
  save_pth = '/home/daewoong/userdata/Study/diffusion/nsynth_diffusion/inference_result_2/'

  epoch = epoch
  #inst = list(inst_dict.values())
  #inst_label = torch.arange(0, len(inst_dict)).to(device)
  inst_label = torch.arange(0, len(inst_dict)).to(device)
  #pitch_label = torch.randint(0, len(pitch_dict), (batch_n, )).to(device)
  #pitch_label = torch.tensor([43 for i in range(batch_n)]).to(device)
  #vel_label = torch.tensor([4 for i in range(batch_n)]).to(device)

  #vel_label = torch.randint(0, len(vel_dict), (batch_n, )).to(device)
  pitch_label = torch.load('/home/daewoong/userdata/Study/diffusion/nsynth_diffusion/test_pitch_list.pth').to(device)
  vel_label = torch.load('/home/daewoong/userdata/Study/diffusion/nsynth_diffusion/test_vel_list.pth').to(device)
  
  y = [inst_label, pitch_label, vel_label]
  
  inst_name_tuple = torch.load('/home/daewoong/userdata/Study/diffusion/nsynth_diffusion/test_genre_list.pth')
  
  output = diffusion.sampling(model = model, batch_n = batch_n, labels = y)
  fig, axs = plt.subplots(1, 2, figsize=(10, 5))


  for i in tqdm(range(batch_n)):
    
    real_pitch = pitch_dict[pitch_label[i].item()]
    real_vel = vel_dict[vel_label[i].item()]
    real_inst_name = inst_dict[inst_label[i].item()]
      
    
    #inst_name_tuple = inst_specific[real_inst_name]
    random_tuple = str(inst_name_tuple[i].item())
    
    if inst_label[i].item() == 4:
      real_inst_name = 'synth_lead_synthetic'


    random_tuple = random_tuple.zfill(3)
    real_pitch = str(real_pitch).zfill(3)
    real_vel = str(real_vel).zfill(3)
    
    
    audio_pth = pth + real_inst_name + '_' + random_tuple + '-' + str(real_pitch) + '-' + str(real_vel) 
    
    org_audio = audio_pth + '.wav'
    vocoder_result = torch.tensor(np.load(org_audio + '.spec.npy'))
    
    vocoder_audio, _ = diffwave_predict(vocoder_result, model_dir)
    torchaudio.save(save_pth + real_inst_name + '_' + random_tuple + '-' + str(real_pitch) + '-' + str(real_vel) + f'_{epoch}_org_vocoder_result.wav' , vocoder_audio.to('cpu'), sample_rate=16000) 
    
    pred_vocoder_audio, _ = diffwave_predict(output[i].squeeze(0).to('cpu'), model_dir)
    
    torchaudio.save(save_pth + real_inst_name + '_' + random_tuple + '-' + str(real_pitch) + '-' + str(real_vel) + f'_{epoch}_diffusion_vocoder_result.wav' , pred_vocoder_audio.to('cpu'), sample_rate=16000) 

    
    axs[0].imshow(vocoder_result, aspect='auto', origin='lower')
    axs[1].imshow(output[i].squeeze(0).to('cpu').numpy(), aspect='auto', origin='lower')

    # 각 subplot에 제목 추가
    axs[0].set_title('Org Result')
    axs[1].set_title('Diffusion Result')

    # 전체 figure에 제목 추가
    fig.suptitle(f'{real_inst_name} - Style: {random_tuple} / Pitch: {real_pitch} / Velocity: {real_vel}')

    # figure 저장
    fig_path = save_pth + f'{epoch}_{real_inst_name}_{random_tuple}-{real_pitch}-{real_vel}_mel_result.png'
    plt.savefig(fig_path, dpi=300)
    # if real_inst_name == 'keyboard_synthetic':
    #   wandb.log({
    #     f'{real_inst_name}_org_audio' : wandb.Audio(org_audio, caption=f"Original Audio, Style : {random_tuple} / Pitch : {real_pitch} / Velocity : {real_vel}"),
    #     f'{real_inst_name}_vocoder result' : wandb.Audio(save_pth + real_inst_name + '_' + random_tuple + '-' + str(real_pitch) + '-' + str(real_vel) + f'_{epoch}_org_vocoder_result.wav', caption="Org Vocoder Result"),
    #     f'{real_inst_name}_pred result' : wandb.Audio(save_pth + real_inst_name + '_' + random_tuple + '-' + str(real_pitch) + '-' + str(real_vel) + f'_{epoch}_diffusion_vocoder_result.wav', caption="Pred Vocoder Result"),
    #     f'{real_inst_name}_comparison' : wandb.Image(fig_path, caption="Mel Result")
    #   })
    
    
    org_audio = wandb.Audio(org_audio, caption="org_audio")
    vocoder_result = wandb.Audio(save_pth + real_inst_name + '_' + random_tuple + '-' + str(real_pitch) + '-' + str(real_vel) + f'_{epoch}_org_vocoder_result.wav')
    pred_result = wandb.Audio(save_pth + real_inst_name + '_' + random_tuple + '-' + str(real_pitch) + '-' + str(real_vel) + f'_{epoch}_diffusion_vocoder_result.wav')
    melspec = wandb.Image(fig_path)
    #org_audio_wav, _ = torchaudio.load(org_audio)
    #vocoder_result_wav, _ = torchaudio.load(save_pth + real_inst_name + '_' + random_tuple + '-' + str(real_pitch) + '-' + str(real_vel) + f'_{epoch}_org_vocoder_result.wav')
    #pred_result_wav, _ = torchaudio.load(save_pth + real_inst_name + '_' + random_tuple + '-' + str(real_pitch) + '-' + str(real_vel) + f'_{epoch}_diffusion_vocoder_result.wav')
    #mel_image = Image.open(fig_path)
    #vocoder_audio_wav, _ = torchaudio.load(save_pth + real_inst_name + '_' + random_tuple + '-' + str(real_pitch) + '-' + str(real_vel) + f'_{epoch}_org_vocoder_result.wav')
    #pred_vocoder_audio_wav, _ = torchaudio.load(save_pth + real_inst_name + '_' + random_tuple + '-' + str(real_pitch) + '-' + str(real_vel) + f'_{epoch}_diffusion_vocoder_result.wav')
    # vocoder_result = wandb.Audio(vocoder_audio_wav, sample_rate=16000)
    # pred_result = wandb.Audio(pred_vocoder_audio_wav, sample_rate=16000)
    
    #mel_spectrogram = wandb.Image(fig_path)
    table.add_data(org_audio, vocoder_result, pred_result, 
                   melspec)
    
  wandb.log({f"Inference_{epoch}_epoch" : table})

    
    
if __name__== "__main__":
  # parser = argparse.ArgumentParser(description="Inference Script")
  # parser.add_argument("--epochs", type=str, required=True, help="pt_epochs")
  
  # args = parser.parse_args()
  
  sys.path.append('/home/daewoong/userdata/Study/diffusion/hifi_gan/diffwave/src/')
  from diffwave.inference import predict as diffwave_predict
  
  dataset = Nsynth()
  inst_dict = dataset.inst_dict
  pitch_dict = dataset.pitch_dict
  vel_dict = dataset.velocity_dict
  inst_specific = dataset.inst_specific
  
  wandb.init(project='Nsynth Diffusion_2')
  wandb.run.name = 'infer'
  epoch_list = [1, 2, 3, 4, 5]
  for i in tqdm(epoch_list):  
    pth_path = f'/home/daewoong/userdata/Study/diffusion/nsynth_diffusion/save_pt_2/model_{i}.pth'
    pt = torch.load(pth_path, map_location='cpu')
    device = 'cuda'
    model = Unet()
    model.load_state_dict(pt)
    print(f'Load {pth_path}')
    model = model.to(device)
    diffusion = Diffusion()
 
      
    columns = ["org_audio", "vocoder_result", "pred_result", "mel_spectrogram"]
    table = wandb.Table(columns=columns)
    
    inference(diffusion, model, inst_dict, pitch_dict, vel_dict, inst_specific, epoch=i)
wandb.finish()
