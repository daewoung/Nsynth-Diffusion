import torch.nn as nn
import torch 
import torchaudio 
import json 
import numpy 
from torch.utils.data import DataLoader

# Train_data : 289205
# Valid_data : 12678

# pitch = 0~127
# velocity = 0~127
# instrument_family_str : {bass : 0 , brass : 1, flute : 2, guitar : 3, keyboard : 4 , mallet : 5, organ : 6, reed : 7, string : 8, synth_lead : 9, vocal : 10}
# spectrogram : (80, 251)

class Nsynth():
  def __init__(self, train_mode = True):
    self.pth = '/home/daewoong/userdata/Study/diffusion/nsynth-train-all/'
    
    if train_mode:
      pth = self.pth + 'examples-train-original.json'
      self.data_dict = json.load(open(pth))
    else:
      pth = self.pth + 'examples-valid-original.json'
      self.data_dict = json.load(open(pth))
      
    self.data_dict_keys = list(self.data_dict.keys())
    
    self.inst_dict = {'bass' : 0 , 'brass' : 1, 'flute' : 2, 'guitar' : 3, 'keyboard' : 4 , 'mallet' : 5, 
                      'organ': 6, 'reed' : 7, 'string' : 8, 'synth_lead' : 9, 'vocal' : 10}
    
  def __len__(self):
    return len(self.data_dict)
  
  def __getitem__(self, idx):
    data_key = self.data_dict_keys[idx]
    spectrogram = self.pth + 'audio/' + data_key + '.wav' + '.spec.npy'
    spectrogram = torch.from_numpy(numpy.load(spectrogram))
    instrument = torch.tensor(self.inst_dict[self.data_dict[data_key]['instrument_family_str']])
    pitch = torch.tensor(int(self.data_dict[data_key]['pitch']))
    velocity = torch.tensor(int(self.data_dict[data_key]['velocity']))
    
    return spectrogram, instrument, pitch, velocity