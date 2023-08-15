import torch.nn as nn
import torch 
import torchaudio 
import json 
import numpy 
from torch.utils.data import DataLoader

'''
# Train_data : 289205
# Valid_data : 12678

pitch = 0~127
velocity = 0~127
instrument_family_str : {bass : 0 , brass : 1, flute : 2, guitar : 3, keyboard : 4 , mallet : 5, organ : 6, reed : 7, string : 8, synth_lead : 9, vocal : 10}
spectrogram : (1, 80, 251)
  sample_rate : 16000
  window_size : 1024
  hop_size : 256
  mel_bins : 80
  fmin : 30
  fmax : 8000
  n_mels : 80


데이터 분석 결과 

velocity : train / valid 데이터의 velocity는 5개만 사용 (25, 50, 75, 100, 127)
pitch : train / valid 데이터의 128개중 112개만 사용
inst : valid엔 synth_lead가 아예 없음. 이유는 예측건데, 다른 악기들에 비해 데이터가 매우 적다 (5501개) 그래서 그러한듯 하다.

'''


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
    
    # self.inst_dict = {'bass' : 0 , 'brass' : 1, 'flute' : 2, 'guitar' : 3, 'keyboard' : 4 , 'mallet' : 5, 
    #                   'organ': 6, 'reed' : 7, 'string' : 8, 'synth_lead' : 9, 'vocal' : 10}
    
    self.inst_dict = {0: 'bass_electronic', 1: 'guitar_synthetic', 2: 'vocal_synthetic',
    3: 'keyboard_acoustic', 4: 'synth_lead', 5: 'keyboard_electronic',
    6: 'organ_acoustic', 7: 'bass_synthetic', 8: 'brass_acoustic',
    9: 'guitar_acoustic', 10: 'guitar_electronic', 11: 'mallet_acoustic',
    12: 'bass_acoustic', 13: 'vocal_electronic', 14: 'string_acoustic',
    15: 'reed_synthetic', 16: 'string_electronic', 17: 'keyboard_synthetic',
    18: 'vocal_acoustic', 19: 'brass_electronic', 20: 'mallet_synthetic',
    21: 'flute_synthetic', 22: 'organ_electronic', 23: 'mallet_electronic',
    24: 'flute_acoustic', 25: 'reed_acoustic', 26: 'flute_electronic',
    27: 'reed_electronic' }

    self.inst_specific = {'guitar_acoustic': ('000', '036'),
                          'bass_synthetic': ('000', '152'),
                          'organ_electronic': ('000', '129'),
                          'guitar_electronic': ('000', '046'),
                          'keyboard_electronic': ('000', '111'),
                          'keyboard_acoustic': ('000', '020'),
                          'vocal_synthetic': ('000', '015'),
                          'string_acoustic': ('000', '090'),
                          'reed_acoustic': ('000', '061'),
                          'flute_acoustic': ('000', '035'),
                          'mallet_electronic': ('000', '013'),
                          'mallet_synthetic': ('000', '004'),
                          'brass_acoustic': ('000', '066'),
                          'guitar_synthetic': ('000', '012'),
                          'flute_synthetic': ('001', '007'),
                          'mallet_acoustic': ('000', '080'),
                          'synth_lead': ('000', '012'),
                          'bass_electronic': ('000', '039'),
                          'keyboard_synthetic': ('001', '012'),
                          'vocal_acoustic': ('001', '029'),
                          'reed_synthetic': ('000', '001'),
                          'organ_acoustic': ('000', '003'),
                          'reed_electronic': ('000', '000'),
                          'vocal_electronic': ('000', '003'),
                          'bass_acoustic': ('000', '000'),
                          'string_electronic': ('000', '001'),
                          'brass_electronic': ('000', '001'),
                          'flute_electronic': ('000', '000')}
    
    self.pitch_dict = {
                        0: 9, 1: 10, 2: 11, 3: 12, 4: 13, 5: 14, 6: 15, 7: 16, 8: 17, 9: 18,
                        10: 19, 11: 20, 12: 21, 13: 22, 14: 23, 15: 24, 16: 25, 17: 26, 18: 27,
                        19: 28, 20: 29, 21: 30, 22: 31, 23: 32, 24: 33, 25: 34, 26: 35, 27: 36,
                        28: 37, 29: 38, 30: 39, 31: 40, 32: 41, 33: 42, 34: 43, 35: 44, 36: 45,
                        37: 46, 38: 47, 39: 48, 40: 49, 41: 50, 42: 51, 43: 52, 44: 53, 45: 54,
                        46: 55, 47: 56, 48: 57, 49: 58, 50: 59, 51: 60, 52: 61, 53: 62, 54: 63,
                        55: 64, 56: 65, 57: 66, 58: 67, 59: 68, 60: 69, 61: 70, 62: 71, 63: 72,
                        64: 73, 65: 74, 66: 75, 67: 76, 68: 77, 69: 78, 70: 79, 71: 80, 72: 81,
                        73: 82, 74: 83, 75: 84, 76: 85, 77: 86, 78: 87, 79: 88, 80: 89, 81: 90,
                        82: 91, 83: 92, 84: 93, 85: 94, 86: 95, 87: 96, 88: 97, 89: 98, 90: 99,
                        91: 100, 92: 101, 93: 102, 94: 103, 95: 104, 96: 105, 97: 106, 98: 107,
                        99: 108, 100: 109, 101: 110, 102: 111, 103: 112, 104: 113, 105: 114,
                        106: 115, 107: 116, 108: 117, 109: 118, 110: 119, 111: 120
                    }
    
    self.velocity_dict = {0 : 25, 1: 50, 2: 75, 3: 100, 4: 127}
    
    self.reverse_inst_dict = {v: k for k, v in self.inst_dict.items()}
    self.reverse_pitch_dict = {v: k for k, v in self.pitch_dict.items()}
    self.reverse_velocity_dict = {v: k for k, v in self.velocity_dict.items()}
    
     
  def __len__(self):
    return len(self.data_dict)
  
  def __getitem__(self, idx):
    data_key = self.data_dict_keys[idx]

    inst = data_key.split('_')[0] + '_' + data_key.split('_')[1]

    spectrogram = self.pth + 'audio/' + data_key + '.wav' + '.spec.npy'
    spectrogram = (torch.from_numpy(numpy.load(spectrogram))).unsqueeze(0)
    instrument = torch.tensor(self.reverse_inst_dict[inst])
    pitch = torch.tensor(self.reverse_pitch_dict[int(self.data_dict[data_key]['pitch'])])
    velocity = torch.tensor(self.reverse_velocity_dict[int(self.data_dict[data_key]['velocity'])])
    
    return spectrogram, instrument, pitch, velocity
  
  

def get_dataloader(batch_size = 16, num_workers = 2):
  train_dataset = Nsynth(train_mode=True)
  valid_dataset = Nsynth(train_mode=False)
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
  valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

  return train_loader, valid_loader  